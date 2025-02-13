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

    bottom_boundary_nodes = locate_dofs_geometrical(V, bottom_boundary)
    top_boundary_nodes_unsorted = locate_dofs_geometrical(V, top_boundary)

    # Order top boundary nodes from left to right
    top_node_x_coords = np.array([mesh.geometry.x[node, 0] for node in top_boundary_nodes_unsorted])
    top_boundary_nodes_sorted_indices = np.argsort(top_node_x_coords)
    top_boundary_nodes = top_boundary_nodes_unsorted[top_boundary_nodes_sorted_indices]

    # Bottom BC: Sinusoidal displacement (Time-dependent)
    amplitude = 5.0
    frequency = 1000   # Hz, approx one cycle in 0.003s
    omega = 2 * np.pi * frequency

    # Define displacement function for bottom boundary - will be updated in time loop
    def bottom_displacement_function(t):
        value = amplitude * np.sin(omega * t)
        return Constant(mesh, np.array([0, value]))
    
    def node_force_function(t, u, v):
        node_force_magnitudes = np.zeros((len(top_boundary_nodes), mesh.geometry.dim))
        sin_value = np.sin(omega * t)
        value_y = - 500000 * sin_value
        node_force_magnitudes[:, 1] = value_y
        node_force_magnitudes[0, 0] = 50000 * sin_value
        node_force_magnitudes[-1, 0] = -50000 * sin_value
        return node_force_magnitudes

    def epsilon(u):
        return 0.5 * (grad(u) + grad(u).T)

    def sigma(u):
        return lmbda * tr(epsilon(u)) * Identity(mesh.geometry.dim) + 2 * mu * epsilon(u)

    u = TrialFunction(V)
    v = TestFunction(V)

    f = Function(V)
    f.x.array[:] = 0.0

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

        node_force_magnitudes = node_force_function(time, u_k.x.array, velocity_k.x.array)
        for i, dof in enumerate(top_boundary_nodes): # Use enumerate to get index 'i'
            node_force = node_force_magnitudes[i] # Force vector at node

            force_vector[2*dof] = node_force[0]
            force_vector[2*dof+1] = node_force[1]

        # Bottom BC: Sinusoidal displacement (Time-dependent)
        bottom_displacement_expr = bottom_displacement_function(time)
        bc_bottom = dirichletbc(bottom_displacement_expr,
                                 bottom_boundary_nodes, V)

        current_bcs = [bc_bottom] # Only bottom BC now, top BC is force

        # Solve linear elasticity problem
        problem = LinearProblem(a, L_body, bcs=current_bcs, u=u_k) # Use pre-assembled RHS b
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
