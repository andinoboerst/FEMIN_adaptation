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
    # W = functionspace(mesh, ("DG", 0, (2, 2)))

    # 3. Material Properties (Linear Elasticity)
    E = 200.0e3  # Young's modulus
    nu = 0.3   # Poisson's ratio
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

    # Bottom BC: Sinusoidal displacement (Time-dependent)
    amplitude = 5.0
    frequency = 1000 # Hz, approx one cycle in 0.003s
    omega = 2 * np.pi * frequency

    # Define displacement function for bottom boundary - will be updated in time loop
    def bottom_displacement_function(t):
        value = amplitude * np.sin(omega * t)
        return Constant(mesh, np.array([0, value]))


    def epsilon(u):
        return sym(grad(u))
        return 0.5 * (grad(u) + grad(u).T)

    def sigma(u):
        return lmbda * tr(epsilon(u)) * Identity(mesh.geometry.dim) + 2 * mu * epsilon(u)

    u = TrialFunction(V)
    v = TestFunction(V)

    f = Function(V)
    f.x.array[:] = 0.0

    a = inner(sigma(u), epsilon(v)) * dx  # Use dx_ufl
    L = inner(f, v) * dx        # Use dx_ufl

    # Time Stepping
    time_total = 3e-3 # s
    dt = 5e-7 # s
    num_steps = int(time_total / dt)
    time = 0.0

    tdim = mesh.topology.dim

    # Initialize solution Function
    u_k = Function(V, name="Displacement")
    u_prev = Function(V) # Function to store displacement from previous time step
    velocity_k = Function(V, name="Velocity") # Function to store velocity
    # sigma_h = Function(W)
    # n = Constant(mesh, np.array([0.0, -1.0]))

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

        # --- Check Displacement Magnitude on Interface Midpoint ---
        # interface_midpoint = np.array([[width / 2.0, 25.0]])  # Midpoint of your interface
        # cell_interface_midpoint = locate_entities(mesh, mesh.topology.dim, interface_midpoint)
        # if len(cell_interface_midpoint) > 0:
        #     cell_index = cell_interface_midpoint[0]
        #     displacement_at_midpoint_value = u_k.eval(cell_index, interface_midpoint[0])
        #     displacement_magnitude = np.linalg.norm(displacement_at_midpoint_value)
        #     print(f"Displacement at Interface Midpoint: {displacement_at_midpoint_value}")
        #     print(f"Magnitude of Displacement at Midpoint: {displacement_magnitude}")
        # else:
        #     print("Midpoint not found in mesh cells.")

        stress_tensor = sigma(u_k)

        interface_tag = 88
        facet_integration_domain = locate_entities(mesh, mesh.topology.dim - 1, interface_boundary)
        facet_tags = meshtags(mesh, mesh.topology.dim - 1, facet_integration_domain, np.full(len(facet_integration_domain), interface_tag, dtype=np.int32))
        ds_interface = Measure("ds", domain=mesh, subdomain_data=facet_tags, subdomain_id=interface_tag)

        n = FacetNormal(mesh)

        # --- 4. Define Unit Vectors (UFL directly, NO PETSc Vec and NO .array) ---
        e_x = as_vector([1.0, 0.0])  # For 2D, x-direction
        e_y = as_vector([0.0, 1.0])  # For 2D, y-direction

        traction_vector_expr = dot(stress_tensor, n)

        # --- 5. Assemble Full Force Components
        force_vector = []
        for e_i in [e_x, e_y]: #, e_z for 3D
            force_integrand = dot(traction_vector_expr, e_i) # Calculate force integrand
            force_component = assemble_scalar(
                form(force_integrand * ds_interface) # Assemble using the pre-calculated integrand
            )
            force_vector.append(force_component)

        force_x, force_y = force_vector #, force_z for 3D if in 3D
        print(f"Force in x-direction: {force_x}")
        print(f"Force in y-direction: {force_y}")

        # --- Calculate Stress and Extract at x=25 (using INITIAL node locations) ---
        # sigma_expr = sigma(u_k)
        # sigma_project = form(inner(sigma_expr, TestFunction(W)) * dx)
        # problem_stress = LinearProblem(form(inner(TestFunction(W), TrialFunction(W)) * dx), sigma_project, u=sigma_h)
        # problem_stress.solve()

        # T_interface = []
        # print(f"\n--- Stress at Nodes initially at x=25 at Time {time:.5f} ---")
        # for node_index in interface_boundary_nodes:
        #     current_coords = mesh.geometry.x[node_index]
        #     cells = locate_entities(mesh, tdim, interface_boundary)
        #     stress_value = sigma_h.eval(current_coords, cells)
        #     sigma_xx = stress_value[0]
        #     sigma_yy = stress_value[3]
        #     sigma_xy = stress_value[1]

        #     t_x = sigma_xx * n.value[0] + sigma_xy * n.value[1]
        #     t_y = sigma_xy * n.value[0] + sigma_yy * n.value[1]

        #     T_interface.append([t_x, t_y])
        
        # print(T_interface)

        # Save DEFORMED MESH and fields
        if step % 100 == 0:
            u_full.append(u_k.x.array.copy())
            v_full.append(velocity_k.x.array.copy())

        # --- Update u_prev for next time step ---
        u_prev.x.array[:] = u_k.x.array[:] # Copy current displacement to u_prev for next velocity calculation

    print("Simulation complete")

    return vtk_mesh(mesh), u_full, v_full, interface_boundary_nodes

    # with open("u_full.npy", "wb") as f:
    #     np.save(f, u_full)

    # with open("v_full.npy", "wb") as f:
    #     np.save(f, v_full)


if __name__ == "__main__":
    tct_elastic_full()