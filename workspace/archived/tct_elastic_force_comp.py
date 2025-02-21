import numpy as np
from mpi4py import MPI
from dolfinx.fem import Constant, Function, functionspace, dirichletbc, locate_dofs_geometrical, assemble_scalar, form
from dolfinx.mesh import create_rectangle, CellType, meshtags, locate_entities
from dolfinx.plot import vtk_mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import TestFunction, TrialFunction, Identity, Measure, grad, inner, dot, tr, dx, FacetNormal

from shared.progress_bar import progressbar  # Assuming this is available


def tct_elastic_generate_f_interface(frequency: int = 1000):
    # 1. Domain and Mesh (same as before)
    width = 100.0
    height = 50.0
    element_size_x = 5.0
    element_size_y = 5.0
    nx = int(width / element_size_x)
    ny = int(height / element_size_y)

    mesh = create_rectangle(MPI.COMM_WORLD, cell_type=CellType.quadrilateral,
                            points=((0.0, 0.0), (width, height)), n=(nx, ny))

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    mesh_connectivity = mesh.topology.connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    # 2. Function Space (same as before)
    V = functionspace(mesh, ("CG", 1, (2,)))
    W = functionspace(mesh, ("CG", 1, (2, 2)))  # Or ("CG", 1) if you want continuous stress
    sigma_projected = Function(W)
    w = TestFunction(W)
    tau = TrialFunction(W)

    # 3. Material Properties (same as before)
    E = 200.0e3
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # 4. Boundary Conditions (Improved)
    def top_boundary(x):
        return np.isclose(x[1], height)

    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)

    def interface_boundary(x):
        return np.isclose(x[1], 25.0)

    def bottom_half(x):
        return x[1] < 25.4

    top_boundary_nodes = locate_dofs_geometrical(V, top_boundary)
    bottom_boundary_nodes = locate_dofs_geometrical(V, bottom_boundary)
    interface_nodes_unsorted = locate_dofs_geometrical(V, interface_boundary)
    bottom_half_nodes = locate_dofs_geometrical(V, bottom_half)

    # bottom_half_node_coords = np.array([mesh.geometry.x[node, 0:2] for node in bottom_half_nodes_unsorted])
    # bottom_half_nodes_sorted_indices = np.argsort(bottom_half_node_coords)
    # bottom_half_nodes = bottom_half_nodes_unsorted[bottom_half_nodes_sorted_indices]
    bottom_half_dofs = np.zeros(len(bottom_half_nodes) * 2, dtype=int)
    for i, node in enumerate(bottom_half_nodes):
        bottom_half_dofs[2 * i: 2 * i + 2] = [node * 2, node * 2 + 1]

    interface_node_x_coords = np.array([mesh.geometry.x[node, 0] for node in interface_nodes_unsorted])
    interface_nodes_sorted_indices = np.argsort(interface_node_x_coords)
    interface_nodes = interface_nodes_unsorted[interface_nodes_sorted_indices]
    interface_dofs = np.zeros(len(interface_nodes) * 2, dtype=int)
    for i, node in enumerate(interface_nodes):
        interface_dofs[2 * i:2 * i + 2] = [node * 2, node * 2 + 1]

    # Define BCs ONCE and update values later
    bc_top_values = np.array([0.0, 0.0], dtype=np.float64)
    bc_top = dirichletbc(bc_top_values, top_boundary_nodes, V)

    # Bottom BC: Sinusoidal displacement (Time-dependent)
    amplitude = 5.0
    omega = 2 * np.pi * frequency

    # Define displacement function for bottom boundary - will be updated in time loop
    def bottom_displacement_function(t):
        value = amplitude * np.sin(omega * t)
        return Constant(mesh, np.array([0, value]))

    # 5. Variational Formulation (same as before)
    def epsilon(u):
        return 0.5 * (grad(u) + grad(u).T)

    def sigma(u):
        return lmbda * tr(epsilon(u)) * Identity(mesh.geometry.dim) + 2 * mu * epsilon(u)

    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(mesh, np.array([0.0, 0.0]))  # Force term
    a = inner(sigma(u), epsilon(v)) * dx
    L = inner(f, v) * dx

    # 6. Time Stepping (Newmark-beta)
    time_total = 3e-3
    dt = 5e-7
    num_steps = int(time_total / dt)
    time = 0.0

    u_k = Function(V, name="Displacement")
    u_prev = Function(V)
    v_k = Function(V, name="Velocity")
    v_prev = Function(V)
    a_k = Function(V, name="Acceleration")
    a_prev = Function(V)

    # Initialize displacement and acceleration to zero
    u_k.x.array[:] = 0.0
    a_prev.x.array[:] = 0.0  # Important initialization

    beta = 0.25  # Newmark-beta parameter
    gamma = 0.5   # Newmark-beta parameter

    # Temporary Functions for predictor step
    u_pred = Function(V)  # Now a Function object
    v_pred = Function(V)  # Now a Function object

    # u_interface = np.zeros((num_steps, len(interface_dofs)))
    u_full = []
    forces = np.zeros((num_steps, len(interface_dofs)))
    for step in progressbar(range(num_steps)):
        time += dt

        bc_bottom = dirichletbc(bottom_displacement_function(time), bottom_boundary_nodes, V)

        # Update current boundary conditions
        current_bcs = [bc_top, bc_bottom]

        # Project the stress to the function space W
        sigma_expr = sigma(u_k)

        # Define the *bilinear* and *linear* forms for the projection
        a_proj = inner(tau, w) * dx  # Bilinear form
        L_proj = inner(sigma_expr, w) * dx  # Linear form (same as bilinear in this L2 projection case)

        problem_stress = LinearProblem(a_proj, L_proj, u=sigma_projected)  # u=sigma_projected sets sigma_projected as the solution
        problem_stress.solve()

        interface_node_forces = np.zeros(len(interface_dofs))

        normal_vector = [0, 1]

        for i, node in enumerate(interface_nodes):
            sigma_xx = sigma_projected.x.array[4 * node]
            sigma_xy = sigma_projected.x.array[4 * node + 1]
            sigma_yy = sigma_projected.x.array[4 * node + 3]

            # Compute the force components (accounting for element size)
            interface_node_forces[2 * i] = sigma_xx * normal_vector[0] + sigma_xy * normal_vector[1] #* element_size_y  # Fx
            interface_node_forces[2 * i + 1] = sigma_xy * normal_vector[0] + sigma_yy * normal_vector[1] #* element_size_y  # Fy


        # traction = np.zeros(len(interface_dofs))

        # for i, (coord, cells) in enumerate(zip(interface_nodes_coords, interface_nodes_cells)):
            # sigma_eval = 0

            # for cell in cells:
            #     sigma_eval += sigma_projected.eval(coord, cell)  # Evaluate at each coordinate in its cell

            # sigma_eval /= len(cells)

            # traction[2 * i] = sigma_eval[0] * 0 - sigma_eval[1] * 5
            # traction[2 * i + 1] = sigma_eval[2] * 0 - sigma_eval[3] * 5


        forces[step, :] = interface_node_forces

        # Solve using Newmark-beta
        # Predictor step (using Function objects)
        u_pred.x.array[:] = u_prev.x.array[:] + dt * v_prev.x.array[:] + 0.5 * dt**2 * (1 - 2 * beta) * a_prev.x.array[:]
        v_pred.x.array[:] = v_prev.x.array[:] + dt * (1 - gamma) * a_prev.x.array[:]

        # Solve for acceleration (using u_pred as initial guess)
        u_k.x.array[:] = u_pred.x.array[:]  # Set initial guess for the solver
        problem = LinearProblem(a, L, bcs=current_bcs, u=u_k)
        u_k = problem.solve()

        # Corrector step
        a_k.x.array[:] = (1 / (beta * dt**2)) * (u_k.x.array[:] - u_pred.x.array[:]) - ((1 - 2 * beta) / (2 * beta)) * a_prev.x.array[:]
        v_k.x.array[:] = v_pred.x.array[:] + dt * gamma * a_k.x.array[:] + dt * (1 - gamma) * a_prev.x.array[:]

        # u_interface[step, :] = u_k.x.array[interface_dofs] - u_prev.x.array[interface_dofs]

        if step % 100 == 0:
            u_full.append(u_k.x.array.copy())

        # Update previous values
        u_prev.x.array[:] = u_k.x.array[:]
        v_prev.x.array[:] = v_k.x.array[:]
        a_prev.x.array[:] = a_k.x.array[:]

    print("Simulation complete")
    return vtk_mesh(mesh), u_full, forces, bottom_half_dofs


def tct_elastic_apply_f_interface(forces_interface, frequency: int = 1000):
    # 1. Domain and Mesh (same as before)
    width = 100.0
    height = 25.0
    element_size_x = 5.0
    element_size_y = 5.0
    nx = int(width / element_size_x)
    ny = int(height / element_size_y)

    mesh = create_rectangle(MPI.COMM_WORLD, cell_type=CellType.quadrilateral,
                            points=((0.0, 0.0), (width, height)), n=(nx, ny))

    # 2. Function Space (same as before)
    V = functionspace(mesh, ("CG", 1, (2,)))
    W = functionspace(mesh, ("CG", 1, (2, 2)))  # Or ("CG", 1) if you want continuous stress
    sigma_projected = Function(W)
    w = TestFunction(W)
    tau = TrialFunction(W)

    # 3. Material Properties (same as before)
    E = 200.0e3
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # 4. Boundary Conditions (Improved)
    def top_boundary(x):
        return np.isclose(x[1], height)

    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)

    top_boundary_nodes_unsorted = locate_dofs_geometrical(V, top_boundary)
    bottom_boundary_nodes = locate_dofs_geometrical(V, bottom_boundary)

    # Order top boundary nodes from left to right
    top_node_x_coords = np.array([mesh.geometry.x[node, 0] for node in top_boundary_nodes_unsorted])
    top_boundary_nodes_sorted_indices = np.argsort(top_node_x_coords)
    top_boundary_nodes = top_boundary_nodes_unsorted[top_boundary_nodes_sorted_indices]
    top_boundary_dofs = np.zeros(len(top_boundary_nodes) * 2, dtype=int)
    for i, node in enumerate(top_boundary_nodes):
        top_boundary_dofs[2 * i:2 * i + 2] = [node * 2, node * 2 + 1]

    facet_indices, facet_markers = [], []
    fdim = mesh.topology.dim - 1
    marker = 88
    facets = locate_entities(mesh, fdim, top_boundary)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

    ds = Measure("ds", domain=mesh, subdomain_data=facet_tag)

    # Bottom BC: Sinusoidal displacement (Time-dependent)
    amplitude = 5.0
    omega = 2 * np.pi * frequency

    # Define displacement function for bottom boundary - will be updated in time loop
    def bottom_displacement_function(t):
        value = amplitude * np.sin(omega * t)
        return Constant(mesh, np.array([0, value]))

    # 5. Variational Formulation (same as before)
    def epsilon(u):
        return 0.5 * (grad(u) + grad(u).T)

    def sigma(u):
        return lmbda * tr(epsilon(u)) * Identity(mesh.geometry.dim) + 2 * mu * epsilon(u)

    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(mesh, np.array([0.0, 0.0]))  # Force term
    a = inner(sigma(u), epsilon(v)) * dx
    L = inner(f, v) * dx

    # 6. Time Stepping (Newmark-beta)
    time_total = 3e-3
    dt = 5e-7
    num_steps = int(time_total / dt)
    time = 0.0

    u_k = Function(V, name="Displacement")
    u_prev = Function(V)
    v_k = Function(V, name="Velocity")
    v_prev = Function(V)
    a_k = Function(V, name="Acceleration")
    a_prev = Function(V)

    neumann_forces = Function(V)

    # Initialize displacement and acceleration to zero
    u_k.x.array[:] = 0.0
    a_prev.x.array[:] = 0.0  # Important initialization

    beta = 0.25  # Newmark-beta parameter
    gamma = 0.5   # Newmark-beta parameter

    # Temporary Functions for predictor step
    u_pred = Function(V)  # Now a Function object
    v_pred = Function(V)  # Now a Function object

    u_full = []
    for step in progressbar(range(num_steps)):
        time += dt

        bc_bottom = dirichletbc(bottom_displacement_function(time), bottom_boundary_nodes, V)

        # Update current boundary conditions
        current_bcs = [bc_bottom]

        neumann_forces.x.array[top_boundary_dofs] = forces_interface[step, :]

        L_neumann = L + dot(neumann_forces, v) * ds(88)

        # Solve using Newmark-beta
        # Predictor step (using Function objects)
        u_pred.x.array[:] = u_prev.x.array[:] + dt * v_prev.x.array[:] + 0.5 * dt**2 * (1 - 2 * beta) * a_prev.x.array[:]
        v_pred.x.array[:] = v_prev.x.array[:] + dt * (1 - gamma) * a_prev.x.array[:]

        # Solve for acceleration (using u_pred as initial guess)
        u_k.x.array[:] = u_pred.x.array[:]  # Set initial guess for the solver
        problem = LinearProblem(a, L_neumann, bcs=current_bcs, u=u_k)
        u_k = problem.solve()

        # Corrector step
        a_k.x.array[:] = (1 / (beta * dt**2)) * (u_k.x.array[:] - u_pred.x.array[:]) - ((1 - 2 * beta) / (2 * beta)) * a_prev.x.array[:]
        v_k.x.array[:] = v_pred.x.array[:] + dt * gamma * a_k.x.array[:] + dt * (1 - gamma) * a_prev.x.array[:]

        if step % 100 == 0:
            u_full.append(u_k.x.array.copy())

        # Update previous values
        u_prev.x.array[:] = u_k.x.array[:]
        v_prev.x.array[:] = v_k.x.array[:]
        a_prev.x.array[:] = a_k.x.array[:]

    print("Simulation complete")
    return vtk_mesh(mesh), u_full


if __name__ == "__main__":
    pass
    # vtk_mesh_obj, u_full_data, v_full_data, interface_nodes = tct_elastic_generate_u_interface()
    # Now you can use vtk_mesh_obj, u_full_data, v_full_data, interface_nodes for post-processing
