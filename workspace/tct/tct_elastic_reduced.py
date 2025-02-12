import numpy as np
from mpi4py import MPI
from dolfinx.fem import Constant, Function, functionspace, dirichletbc, locate_dofs_geometrical, locate_dofs_topological
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary, locate_entities, meshtags
from dolfinx.plot import vtk_mesh
# from dolfinx.fem.petsc import LinearProblem
from misc.my_petsc import LinearProblem
from ufl import TestFunction, TrialFunction, Identity, Measure, grad, inner, tr, dx

from misc.progress_bar import progressbar


def tct_elastic_reduced() -> None:
    # 1. Domain and Mesh
    width = 100.0
    height = 25.0
    element_size_x = 5.0
    element_size_y = 5.0
    nx = int(width / element_size_x)
    ny = int(height / element_size_y)

    mesh = create_rectangle(MPI.COMM_WORLD, cell_type=CellType.quadrilateral,
                             points=((0.0, 0.0), (width, height)), n=(nx, ny))

    # 2. Function Space
    V = functionspace(mesh, ("CG", 1, (2,)))

    # 3. Material Properties (Linear Elasticity)
    E = 200.0e3  # Young's modulus
    nu = 0.3  # Poisson's ratio
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # 4. Boundary Conditions
    def top_boundary(x):
        return np.isclose(x[1], height)

    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)
    
    boundaries = [bottom_boundary, top_boundary]

    facet_indices, facet_markers = [], []
    fdim = mesh.topology.dim - 1
    for (marker, locator) in enumerate(boundaries):
        facets = locate_entities(mesh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

    bottom_boundary_nodes = locate_dofs_geometrical(V, bottom_boundary)
    top_boundary_nodes = locate_dofs_geometrical(V, top_boundary)

    # Define measure for top boundary facets
    # top_boundary_facets = locate_entities_boundary(mesh, mesh.topology.dim-1, top_boundary) # Use locate_entities_boundary
    # ds_top = Measure("ds", domain=mesh, subdomain_data=top_boundary_facets) # Measure on top boundary

    # Bottom BC: Sinusoidal displacement (Time-dependent)
    amplitude = 5.0
    frequency = 1000   # Hz, approx one cycle in 0.003s
    omega = 2 * np.pi * frequency

    # Define displacement function for bottom boundary - will be updated in time loop
    def bottom_displacement_function(t):
        value = amplitude * np.sin(omega * t)
        return Constant(mesh, np.array([0, value]))
    
    def node_force_function(n, u, v):
        node_force_magnitudes = np.zeros((len(top_boundary_nodes), mesh.geometry.dim))
        if n > 3000:
            g = -1
        else:
            g = 1
        for i in range(len(top_boundary_nodes)):
            node_force_magnitudes[i, :] = [100000.0 * g, 5000.0]
        return node_force_magnitudes
    
    # def traction_function(t):
    #     def traction_function_internal(x):
    #         print(x)
    #         traction_magnitude = 1000 + (t * 100)
    #         traction_values = np.vstack((7 * x[1], 3 * x[0]))
    #         return traction_values
    #     return traction_function_internal

    def epsilon(u):
        return 0.5 * (grad(u) + grad(u).T)

    def sigma(u):
        return lmbda * tr(epsilon(u)) * Identity(mesh.geometry.dim) + 2 * mu * epsilon(u)

    u = TrialFunction(V)
    v = TestFunction(V)

    f = Function(V)
    f.x.array[:] = 0.0

    # traction_expr = Function(V)

    a = inner(sigma(u), epsilon(v)) * dx   # Use dx_ufl
    L_body = inner(f, v) * dx      # Use dx_ufl, body force part of L


    # Time Stepping
    time_total = 3e-3   # s
    dt = 5e-7     # s
    num_steps = int(time_total / dt)
    time = 0.0

    # Initialize solution Function
    u_k = Function(V, name="Displacement")
    u_prev = Function(V)   # Function to store displacement from previous time step
    velocity_k = Function(V, name="Velocity")   # Function to store velocity

    # Initialize u_prev to zero
    u_prev.x.array[:] = 0.0

    u_full = []
    v_full = []
    for step in progressbar(range(num_steps)):
        time += dt

        # --- Node-dependent top boundary force ---
        force_vector = np.zeros(V.dofmap.index_map.size_global * V.dofmap.index_map_bs, dtype=np.float64) # Global force vector

        node_force_magnitudes = node_force_function(step, u_k.x.array, velocity_k.x.array)
        for i, dof in enumerate(top_boundary_nodes): # Use enumerate to get index 'i'
            node_force = node_force_magnitudes[i] # Force vector at node

            # Add force to the global force vector at the correct DOF indices
            global_dof_index_local = V.dofmap.index_map.local_to_global(np.array([dof], dtype=np.int32)) # Get global DOF index for the current node
            dof_indices = np.array([global_dof_index_local]) # Wrap in numpy array for consistency
            for j in range(V.dofmap.index_map_bs): # Iterate over block size (components of vector function space)
                global_dof_index = dof_indices[0] * V.dofmap.index_map_bs + j # Calculate global index
                local_range = V.dofmap.index_map.local_range
                if local_range[0] <= global_dof_index < local_range[1]: # Check if current process owns this DOF
                    local_dof_index = global_dof_index - local_range[0] # Calculate local index based on global index and local range
                    force_vector[local_dof_index] += node_force[j] # Add force component


        # --- Neumann term in linear form ---
        # traction_expr.interpolate(traction_function(time))
        # L_traction = inner(traction_expr, v) * ds_top  # Surface traction integral on top boundary
        # L = L_body + L_traction # Total linear form is body force + traction
        L = L_body

        # Bottom BC: Sinusoidal displacement (Time-dependent)
        bottom_displacement_expr = bottom_displacement_function(time)
        bc_bottom = dirichletbc(bottom_displacement_expr,
                                 bottom_boundary_nodes, V)

        current_bcs = [bc_bottom] # Only bottom BC now, top BC is force

        # Solve linear elasticity problem
        problem = LinearProblem(a, L, bcs=current_bcs, u=u_k) # Use pre-assembled RHS b
        u_k = problem.solve(force_vector)

        velocity_k.x.array[:] = (u_k.x.array[:] - u_prev.x.array[:]) / dt

        # Save DEFORMED MESH and fields
        if step % 100 == 0:
            u_full.append(u_k.x.array.copy())
            v_full.append(velocity_k.x.array.copy())

        # --- Update u_prev for next time step ---
        u_prev.x.array[:] = u_k.x.array[:]   # Copy current displacement to u_prev for next velocity calculation

    print("Simulation complete")

    return vtk_mesh(mesh), u_full, v_full


if __name__ == "__main__":
    vtk_m, u_f, v_f = tct_elastic_reduced()
