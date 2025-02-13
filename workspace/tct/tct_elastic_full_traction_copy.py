import numpy as np
import pickle

# import dolfinx
from mpi4py import MPI
from dolfinx.fem import Constant, Function, functionspace, dirichletbc, locate_dofs_geometrical, form, assemble_scalar
from dolfinx.mesh import create_rectangle, CellType, locate_entities, meshtags
from dolfinx.plot import vtk_mesh
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
from ufl import TestFunction, TrialFunction, Identity, FacetNormal, Measure, grad, inner, tr, dx, dot, as_vector, sym, rank

from misc.progress_bar import progressbar


def tct_elastic_full() -> None:
    # 1. Domain and Mesh
    width = 100.0
    height = 50.0
    element_size_x = 5.0
    element_size_y = 5.0
    nx = int(width / element_size_x)
    ny = int(height / element_size_y)

    mesh = create_rectangle(MPI.COMM_WORLD, cell_type=CellType.quadrilateral,
                            points=((0.0, 0.0), (width, height)), n=(nx, ny))

    # 2. Function Space
    V = functionspace(mesh, ("CG", 1, (2,)))
    W = functionspace(mesh, ("DG", 0, (2, 2))) # DG space for stress tensor (rank-2 in 2D, so shape (2,2))
    Wvec = functionspace(mesh, ("DG", 0, (2,)))


    # 3. Material Properties (Linear Elasticity)
    E = 200.0e3  # Young's modulus
    nu = 0.3    # Poisson's ratio
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # 4. Boundary Conditions
    def top_boundary(x):
        return np.isclose(x[1], height)

    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)

    def interface_boundary(x):
        return np.isclose(x[1], 25.0)

    top_boundary_nodes = locate_dofs_geometrical(V, top_boundary)
    bottom_boundary_nodes = locate_dofs_geometrical(V, bottom_boundary)
    interface_boundary_nodes_unsorted = locate_dofs_geometrical(V, interface_boundary)

    # Order interface boundary nodes from left to right
    interface_node_x_coords = np.array([mesh.geometry.x[node, 0] for node in interface_boundary_nodes_unsorted])
    interface_boundary_nodes_sorted_indices = np.argsort(interface_node_x_coords)
    interface_boundary_nodes = interface_boundary_nodes_unsorted[interface_boundary_nodes_sorted_indices]

    # Top BC: Fixed (Dirichlet BC)
    bc_top = dirichletbc(np.array([0.0, 0.0], dtype=np.float64), top_boundary_nodes, V) # Use top_boundary function

    # Bottom BC: Constant displacement (Static case for debugging)
    amplitude = 5.0


    # Define displacement function for bottom boundary - will be updated in time loop
    def bottom_displacement_function(t): # t argument still needed but not used now
        value = amplitude # Constant amplitude
        return Constant(mesh, np.array([0, value]))

    def epsilon(u):
        return sym(grad(u))

    def sigma(u):
        return lmbda * tr(epsilon(u)) * Identity(mesh.geometry.dim) + 2 * mu * epsilon(u)

    u = TrialFunction(V)
    v = TestFunction(V)

    f = Function(V)
    f.x.array[:] = 0.0

    a = inner(sigma(u), epsilon(v)) * dx   # Use dx_ufl
    L = inner(f, v) * dx          # Use dx_ufl

    # Time Stepping - SET TO 1 STEP FOR STATIC DEBUGGING
    time_total = 3e-3 # s
    dt = 5e-7 # s
    num_steps = int(time_total / dt)
    time = 0.0

    tdim = mesh.topology.dim

    # Initialize solution Function
    u_k = Function(V, name="Displacement")
    u_prev = Function(V) # Function to store displacement from previous time step
    velocity_k = Function(V, name="Velocity") # Function to store velocity


    # Initialize u_prev to zero
    u_prev.x.array[:] = 0.0

    u_full = []
    v_full = []

    for step in progressbar(range(num_steps)):
        time += dt

        # Update top boundary condition
        bc_top = dirichletbc(np.array([0.0, 0.0], dtype=np.float64), top_boundary_nodes, V)

        # Update bottom boundary condition
        bottom_displacement_expr = bottom_displacement_function(time)

        bc_bottom = dirichletbc(bottom_displacement_expr, # Use u_bottom_func again
                                         bottom_boundary_nodes, V)

        current_bcs = [bc_top, bc_bottom]

        # Solve linear elasticity problem
        problem = LinearProblem(a, L, bcs=current_bcs, u=u_k)
        u_k = problem.solve()

        velocity_k.x.array[:] = (u_k.x.array[:] - u_prev.x.array[:]) / dt

        stress_tensor = sigma(u_k)

        # --- Project stress_tensor into Function Space W ---
        stress_h = Function(W, name="Stress") # Function to hold projected stress
        sigma_project = form(inner(stress_tensor, TestFunction(W)) * dx) # Form for projection
        a_proj = form(inner(TrialFunction(W), TestFunction(W)) * dx) # Mass matrix form for projection
        problem_stress = LinearProblem(a_proj, sigma_project, u=stress_h) # Linear problem for projection
        problem_stress.solve() # Solve to get stress_h as a dolfinx Function

        print(stress_h.x.array)

        interface_tag = 88
        facet_integration_domain = locate_entities(mesh, mesh.topology.dim - 1, interface_boundary)
        facet_tags = meshtags(mesh, mesh.topology.dim - 1, facet_integration_domain, np.full(len(facet_integration_domain), interface_tag, dtype=np.int32))
        ds_interface = Measure("ds", domain=mesh, subdomain_data=facet_tags, subdomain_id=interface_tag)


        n = FacetNormal(mesh)
        traction_vector_expr = dot(stress_h, n) # Using PROJECTED stress_h

        # --- Project traction_vector_expr into Function Space Wvec ---
        traction_h = Function(Wvec, name="Traction") # Function to hold projected traction vector
        traction_project = form(inner(traction_vector_expr, TestFunction(Wvec)) * dx) # Form for projection
        a_proj_traction = form(inner(TrialFunction(Wvec), TestFunction(Wvec)) * dx) # Mass matrix form for projection
        problem_traction = LinearProblem(a_proj_traction, traction_project, u=traction_h) # Linear problem for projection
        problem_traction.solve() # Solve to get traction_h as a dolfinx Function

        # Now use traction_h for evaluation, not traction_vector_expr (which is a UFL expression)

        # --- Calculate Traction Vector at Interface Nodes ---
        interface_node_coords = mesh.geometry.x[interface_boundary_nodes] # Coordinates of interface nodes
        nodal_traction_vectors = [] # List to store traction vectors at each node

        print("\n--- Traction Vectors at Interface Nodes ---")
        for node_index, node_coord in zip(interface_boundary_nodes, interface_node_coords):
            node_traction_vector = traction_h.eval(node_coord, np.array([0])) # Evaluate PROJECTED traction (traction_h)
            nodal_traction_vectors.append(node_traction_vector)
            print(f"Node Index: {node_index}, Coordinates: {node_coord}, Traction Vector: {node_traction_vector}")
        print("--- End Traction Vectors at Interface Nodes ---")


        # --- 4. Define Unit Vectors ---
        e_x = as_vector([1.0, 0.0])
        e_y = as_vector([0.0, 1.0])

        # --- 5. Assemble Full Force Components (Using explicit traction vector - TOTAL FORCE) ---
        force_vector = []
        for e_i in [e_x, e_y]: #, e_z for 3D
            force_integrand = dot(traction_vector_expr, e_i) # Calculate force integrand
            force_component = assemble_scalar(
                form(force_integrand * ds_interface) # Assemble using the pre-calculated integrand
            )
            force_vector.append(force_component)

        force_x, force_y = force_vector #, force_z for 3D if in 3D
        print(f"Force in x-direction (Total Integrated Force): {force_x}")
        print(f"Force in y-direction (Total Integrated Force): {force_y}")

        # --- Check Displacement Magnitude on Interface Midpoint ---
        interface_x_mid = 50.0
        interface_y_mid = 25.0


        def midpoint_marker(x):
            is_midpoint_x = np.isclose(x[0], interface_x_mid, atol=element_size_x/2.0)
            is_midpoint_y = np.isclose(x[1], interface_y_mid, atol=element_size_x/2.0)
            return np.logical_and(is_midpoint_x, is_midpoint_y)

        midpoint_index = locate_dofs_geometrical(V, midpoint_marker)

        if len(midpoint_index) > 0:
            point_index = midpoint_index[0]
            u_x = u_k.x.array[::2][point_index]
            u_y = u_k.x.array[1::2][point_index]
            print(u_y)
            displacement_magnitude = np.linalg.norm([u_x, u_y])
            print(f"Magnitude of Displacement at Midpoint: {displacement_magnitude}")
        else:
            print("Midpoint not found in mesh cells.")


        # Save DEFORMED MESH and fields
        if step % 100 == 0:
            u_full.append(u_k.x.array.copy())
            v_full.append(velocity_k.x.array.copy())

        # --- Update u_prev for next time step ---
        u_prev.x.array[:] = u_k.x.array[:] # Copy current displacement to u_prev for next velocity calculation

    print("Simulation complete")

    return vtk_mesh(mesh), u_full, v_full, interface_boundary_nodes


if __name__ == "__main__":
    tct_elastic_full()