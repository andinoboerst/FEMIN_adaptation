import numpy as np
from mpi4py import MPI
from dolfinx.fem import Constant, Function, functionspace, dirichletbc, locate_dofs_geometrical, assemble_scalar, form
from dolfinx.mesh import create_rectangle, CellType, meshtags, locate_entities
from dolfinx.plot import vtk_mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import TestFunction, TrialFunction, Identity, Measure, grad, inner, dot, tr, dx, FacetNormal

from shared.progress_bar import progressbar  # Assuming this is available


def tct_elastic_generate(frequency: int = 1000):
    # 1. Domain and Mesh (same as before)
    width = 100.0
    height = 50.0
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

    def interface_boundary(x):
        return np.isclose(x[1], 25.0)

    top_boundary_nodes = locate_dofs_geometrical(V, top_boundary)
    bottom_boundary_nodes = locate_dofs_geometrical(V, bottom_boundary)
    interface_nodes_unsorted = locate_dofs_geometrical(V, interface_boundary)  # No sorting needed, we constrain both x and y

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
    # time_total = 3e-3
    time_total = 1e-3
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

    u_interface = np.zeros((num_steps, len(interface_dofs)))
    f_interface = np.zeros((num_steps, len(interface_dofs)))
    f_interface_prev = np.zeros(len(interface_dofs))
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


        f_interface[step, :] = interface_node_forces - f_interface_prev
        f_interface_prev = interface_node_forces

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

        u_interface[step, :] = u_k.x.array[interface_dofs] - u_prev.x.array[interface_dofs]

        # Update previous values
        u_prev.x.array[:] = u_k.x.array[:]
        v_prev.x.array[:] = v_k.x.array[:]
        a_prev.x.array[:] = a_k.x.array[:]

    print("Simulation complete")
    return f_interface, u_interface


def tct_elastic_apply(predictor, frequency: int = 1000):
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
    # time_total = 3e-3
    time_total = 1e-3
    dt = 5e-7
    num_steps = int(time_total / dt)
    time = 0.0

    u_k = Function(V, name="Displacement")
    u_prev = Function(V)
    v_k = Function(V, name="Velocity")
    v_prev = Function(V)
    a_k = Function(V, name="Acceleration")
    a_prev = Function(V)

    u_top = Function(V, name="InterfaceBC")

    # Initialize displacement and acceleration to zero
    u_k.x.array[:] = 0.0
    a_prev.x.array[:] = 0.0  # Important initialization

    beta = 0.25  # Newmark-beta parameter
    gamma = 0.5   # Newmark-beta parameter

    # Temporary Functions for predictor step
    u_pred = Function(V)  # Now a Function object
    v_pred = Function(V)  # Now a Function object

    u_full = []
    v_full = []
    f_prev = np.zeros(len(top_boundary_dofs))
    for step in progressbar(range(num_steps)):
        time += dt

        # Project the stress to the function space W
        sigma_expr = sigma(u_k)

        # Define the *bilinear* and *linear* forms for the projection
        a_proj = inner(tau, w) * dx  # Bilinear form
        L_proj = inner(sigma_expr, w) * dx  # Linear form (same as bilinear in this L2 projection case)

        problem_stress = LinearProblem(a_proj, L_proj, u=sigma_projected)  # u=sigma_projected sets sigma_projected as the solution
        problem_stress.solve()

        interface_node_forces = np.zeros(len(top_boundary_dofs))

        normal_vector = [0, 1]

        for i, node in enumerate(top_boundary_nodes):
            sigma_xx = sigma_projected.x.array[4 * node]
            sigma_xy = sigma_projected.x.array[4 * node + 1]
            sigma_yy = sigma_projected.x.array[4 * node + 3]

            # Compute the force components (accounting for element size)
            interface_node_forces[2 * i] = sigma_xx * normal_vector[0] + sigma_xy * normal_vector[1] #* element_size_y  # Fx
            interface_node_forces[2 * i + 1] = sigma_xy * normal_vector[0] + sigma_yy * normal_vector[1] #* element_size_y  # Fy

        delta_f = interface_node_forces - f_prev
        f_prev = interface_node_forces

        u_top.x.array[top_boundary_dofs] = u_prev.x.array[top_boundary_dofs] + predictor.predict([delta_f])[0]
        bc_top = dirichletbc(u_top, top_boundary_nodes)

        bc_bottom = dirichletbc(bottom_displacement_function(time), bottom_boundary_nodes, V)

        # Update current boundary conditions
        current_bcs = [bc_bottom, bc_top]

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

        if step % 100 == 0:
            u_full.append(u_k.x.array.copy())
            v_full.append(v_k.x.array.copy())

        # Update previous values
        u_prev.x.array[:] = u_k.x.array[:]
        v_prev.x.array[:] = v_k.x.array[:]
        a_prev.x.array[:] = a_k.x.array[:]

    print("Simulation complete")
    return vtk_mesh(mesh), u_full, v_full


def tct_elastic_predictor_error_comparison(predictor, frequency: int = 1000):
    # 1. Domain and Mesh (same as before)
    width = 100.0
    height_pred = 25.0
    height_real = 50.0
    element_size_x = 5.0
    element_size_y = 5.0
    nx = int(width / element_size_x)
    ny_pred = int(height_pred / element_size_y)
    ny_real = int(height_real / element_size_y)

    mesh_pred = create_rectangle(MPI.COMM_WORLD, cell_type=CellType.quadrilateral,
                            points=((0.0, 0.0), (width, height_pred)), n=(nx, ny_pred))

    mesh_real = create_rectangle(MPI.COMM_WORLD, cell_type=CellType.quadrilateral,
                            points=((0.0, 0.0), (width, height_real)), n=(nx, ny_real))

    # 2. Function Space (same as before)
    V_pred = functionspace(mesh_pred, ("CG", 1, (2,)))
    W_pred = functionspace(mesh_pred, ("DG", 0, (2, 2)))  # Or ("CG", 1) if you want continuous stress
    sigma_projected_pred = Function(W_pred)
    w_pred = TestFunction(W_pred)
    tau_pred = TrialFunction(W_pred)

    V_real = functionspace(mesh_real, ("CG", 1, (2,)))

    # 3. Material Properties (same as before)
    E = 200.0e3
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # 4. Boundary Conditions (Improved)
    def top_boundary_pred(x):
        return np.isclose(x[1], height_pred)
    
    def top_boundary_real(x):
        return np.isclose(x[1], height_real)

    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)

    def bottom_half_real(x):
        return x[1] <= 25.4

    top_boundary_nodes_unsorted_pred = locate_dofs_geometrical(V_pred, top_boundary_pred)
    bottom_boundary_nodes_pred = locate_dofs_geometrical(V_pred, bottom_boundary)

    # Order top boundary nodes from left to right
    top_node_x_coords_pred = np.array([mesh_pred.geometry.x[node, 0] for node in top_boundary_nodes_unsorted_pred])
    top_boundary_nodes_sorted_indices_pred = np.argsort(top_node_x_coords_pred)
    top_boundary_nodes_pred = top_boundary_nodes_unsorted_pred[top_boundary_nodes_sorted_indices_pred]
    top_boundary_dofs_pred = np.zeros(len(top_boundary_nodes_pred) * 2, dtype=int)
    for i, node in enumerate(top_boundary_nodes_pred):
        top_boundary_dofs_pred[2 * i:2 * i + 2] = [node * 2, node * 2 + 1]

    def interface_boundary(x):
        return np.isclose(x[1], 25.0)

    top_boundary_nodes_real = locate_dofs_geometrical(V_real, top_boundary_real)
    bottom_boundary_nodes_real = locate_dofs_geometrical(V_real, bottom_boundary)
    interface_nodes_unsorted_real = locate_dofs_geometrical(V_real, interface_boundary)
    bottom_half_nodes_real = locate_dofs_geometrical(V_real, bottom_half_real)

    interface_node_x_coords_real = np.array([mesh_real.geometry.x[node, 0] for node in interface_nodes_unsorted_real])
    interface_nodes_sorted_indices_real = np.argsort(interface_node_x_coords_real)
    interface_nodes_real = interface_nodes_unsorted_real[interface_nodes_sorted_indices_real]
    interface_dofs_real = np.zeros(len(interface_nodes_real) * 2, dtype=int)
    for i, node in enumerate(interface_nodes_real):
        interface_dofs_real[2 * i:2 * i + 2] = [node * 2, node * 2 + 1]

    # Define BCs ONCE and update values later
    bc_top_values_real = np.array([0.0, 0.0], dtype=np.float64)
    bc_top_real = dirichletbc(bc_top_values_real, top_boundary_nodes_real, V_real)

    # Bottom BC: Sinusoidal displacement (Time-dependent)
    amplitude = 5.0
    omega = 2 * np.pi * frequency

    # Define displacement function for bottom boundary - will be updated in time loop
    def bottom_displacement_function(t):
        value = amplitude * np.sin(omega * t)
        return Constant(mesh_pred, np.array([0, value]))

    # 5. Variational Formulation (same as before)
    def epsilon(u):
        return 0.5 * (grad(u) + grad(u).T)

    def sigma(u):
        return lmbda * tr(epsilon(u)) * Identity(mesh_pred.geometry.dim) + 2 * mu * epsilon(u)

    u_pred = TrialFunction(V_pred)
    v_pred = TestFunction(V_pred)
    f_pred = Constant(mesh_pred, np.array([0.0, 0.0]))  # Force term
    a_pred = inner(sigma(u_pred), epsilon(v_pred)) * dx
    L_pred = inner(f_pred, v_pred) * dx

    u_real = TrialFunction(V_real)
    v_real = TestFunction(V_real)
    f_real = Constant(mesh_real, np.array([0.0, 0.0]))  # Force term
    a_real = inner(sigma(u_real), epsilon(v_real)) * dx
    L_real = inner(f_real, v_real) * dx

    # 6. Time Stepping (Newmark-beta)
    # time_total = 3e-3
    time_total = 1e-3
    dt = 5e-7
    num_steps = int(time_total / dt)
    time = 0.0

    u_k_pred = Function(V_pred, name="Displacement")
    u_prev_pred = Function(V_pred)
    v_k_pred = Function(V_pred, name="Velocity")
    v_prev_pred = Function(V_pred)
    a_k_pred = Function(V_pred, name="Acceleration")
    a_prev_pred = Function(V_pred)

    u_top_pred = Function(V_pred, name="InterfaceBC")

    # Initialize displacement and acceleration to zero
    u_k_pred.x.array[:] = 0.0
    a_prev_pred.x.array[:] = 0.0  # Important initialization

    u_k_real = Function(V_real, name="Displacement")
    u_prev_real = Function(V_real)
    v_k_real = Function(V_real, name="Velocity")
    v_prev_real = Function(V_real)
    a_k_real = Function(V_real, name="Acceleration")
    a_prev_real = Function(V_real)

    # Initialize displacement and acceleration to zero
    u_k_real.x.array[:] = 0.0
    a_prev_real.x.array[:] = 0.0  # Important initialization

    beta = 0.25  # Newmark-beta parameter
    gamma = 0.5   # Newmark-beta parameter

    # Temporary Functions for predictor step
    u_temp_pred = Function(V_pred)  # Now a Function object
    v_temp_pred = Function(V_pred)  # Now a Function object

    u_temp_real = Function(V_real)  # Now a Function object
    v_temp_real = Function(V_real)  # Now a Function object

    u_full_pred = []
    v_full_pred = []
    u_full_real = []
    v_full_real = []
    prediction_error = []
    f_prev_pred = np.zeros(len(top_boundary_dofs_pred))
    for step in progressbar(range(num_steps)):
        time += dt

        bc_bottom_pred = dirichletbc(bottom_displacement_function(time), bottom_boundary_nodes_pred, V_pred)

        # Project the stress to the function space W
        sigma_expr_pred = sigma(u_k_pred)

        # Define the *bilinear* and *linear* forms for the projection
        a_proj_pred = inner(tau_pred, w_pred) * dx  # Bilinear form
        L_proj_pred = inner(sigma_expr_pred, w_pred) * dx  # Linear form (same as bilinear in this L2 projection case)

        problem_stress_pred = LinearProblem(a_proj_pred, L_proj_pred, u=sigma_projected_pred)  # u=sigma_projected sets sigma_projected as the solution
        problem_stress_pred.solve()

        interface_node_forces_pred = np.zeros(len(top_boundary_dofs_pred))

        normal_vector = [0, 1]

        for i, node in enumerate(top_boundary_nodes_pred):
            sigma_xx = sigma_projected_pred.x.array[4 * node]
            sigma_xy = sigma_projected_pred.x.array[4 * node + 1]
            sigma_yy = sigma_projected_pred.x.array[4 * node + 3]

            # Compute the force components (accounting for element size)
            interface_node_forces_pred[2 * i] = sigma_xx * normal_vector[0] + sigma_xy * normal_vector[1] #* element_size_y  # Fx
            interface_node_forces_pred[2 * i + 1] = sigma_xy * normal_vector[0] + sigma_yy * normal_vector[1] #* element_size_y  # Fy

        delta_f_pred = interface_node_forces_pred - f_prev_pred
        f_prev_pred = interface_node_forces_pred

        u_top_pred.x.array[top_boundary_dofs_pred] = u_k_pred.x.array[top_boundary_dofs_pred] + predictor.predict([delta_f_pred])[0]
        bc_top_pred = dirichletbc(u_top_pred, top_boundary_nodes_pred)

        # Update current boundary conditions
        current_bcs_pred = [bc_bottom_pred, bc_top_pred]

        # Solve using Newmark-beta
        # Predictor step (using Function objects)
        u_temp_pred.x.array[:] = u_prev_pred.x.array[:] + dt * v_prev_pred.x.array[:] + 0.5 * dt**2 * (1 - 2 * beta) * a_prev_pred.x.array[:]
        v_temp_pred.x.array[:] = v_prev_pred.x.array[:] + dt * (1 - gamma) * a_prev_pred.x.array[:]

        # Solve for acceleration (using u_pred as initial guess)
        u_k_pred.x.array[:] = u_temp_pred.x.array[:]  # Set initial guess for the solver
        problem_real = LinearProblem(a_pred, L_pred, bcs=current_bcs_pred, u=u_k_pred)
        u_k_pred = problem_real.solve()

        # Corrector step
        a_k_pred.x.array[:] = (1 / (beta * dt**2)) * (u_k_pred.x.array[:] - u_temp_pred.x.array[:]) - ((1 - 2 * beta) / (2 * beta)) * a_prev_pred.x.array[:]
        v_k_pred.x.array[:] = v_temp_pred.x.array[:] + dt * gamma * a_k_pred.x.array[:] + dt * (1 - gamma) * a_prev_pred.x.array[:]

        # Update previous values
        u_prev_pred.x.array[:] = u_k_pred.x.array[:]
        v_prev_pred.x.array[:] = v_k_pred.x.array[:]
        a_prev_pred.x.array[:] = a_k_pred.x.array[:]


        bc_bottom_real = dirichletbc(bottom_displacement_function(time), bottom_boundary_nodes_real, V_real)

        current_bcs = [bc_top_real, bc_bottom_real]

        # Solve using Newmark-beta
        # Predictor step (using Function objects)
        u_temp_real.x.array[:] = u_prev_real.x.array[:] + dt * v_prev_real.x.array[:] + 0.5 * dt**2 * (1 - 2 * beta) * a_prev_real.x.array[:]
        v_temp_real.x.array[:] = v_prev_real.x.array[:] + dt * (1 - gamma) * a_prev_real.x.array[:]

        # Solve for acceleration (using u_pred as initial guess)
        u_k_real.x.array[:] = u_temp_real.x.array[:]  # Set initial guess for the solver
        problem_real = LinearProblem(a_real, L_real, bcs=current_bcs, u=u_k_real)
        u_k_real = problem_real.solve()

        # Corrector step
        a_k_real.x.array[:] = (1 / (beta * dt**2)) * (u_k_real.x.array[:] - u_temp_real.x.array[:]) - ((1 - 2 * beta) / (2 * beta)) * a_prev_real.x.array[:]
        v_k_real.x.array[:] = v_temp_real.x.array[:] + dt * gamma * a_k_real.x.array[:] + dt * (1 - gamma) * a_prev_real.x.array[:]

        # Update previous values
        u_prev_real.x.array[:] = u_k_real.x.array[:]
        v_prev_real.x.array[:] = v_k_real.x.array[:]
        a_prev_real.x.array[:] = a_k_real.x.array[:]

        prediction_error.append(abs(u_k_real.x.array[interface_dofs_real] - u_k_pred.x.array[top_boundary_dofs_pred]).mean())


        if step % 100 == 0:
            u_full_pred.append(u_k_pred.x.array.copy())
            v_full_pred.append(v_k_pred.x.array.copy())
            u_full_real.append(u_k_real.x.array.copy())
            v_full_real.append(v_k_real.x.array.copy())

    print("Simulation complete")
    return vtk_mesh(mesh_pred), u_full_pred, v_full_pred, vtk_mesh(mesh_real), u_full_real, v_full_real, prediction_error, bottom_half_nodes_real


if __name__ == "__main__":
    vtk_mesh_obj, u_full_data, v_full_data, interface_nodes = tct_elastic_generate()
    # Now you can use vtk_mesh_obj, u_full_data, v_full_data, interface_nodes for post-processing
