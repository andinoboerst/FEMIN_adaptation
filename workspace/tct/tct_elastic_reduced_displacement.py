import numpy as np
from mpi4py import MPI
from dolfinx.fem import Constant, Function, functionspace, dirichletbc, locate_dofs_geometrical
from dolfinx.mesh import create_rectangle, CellType
from dolfinx.plot import vtk_mesh
from ufl import TestFunction, TrialFunction, Identity, grad, inner, tr, dx
from dolfinx.fem.petsc import LinearProblem

from misc.progress_bar import progressbar  # Assuming this is available


class Predictor:
    def __init__(self, omega, shape) -> None:
        self.omega = omega
        self.shape = shape

        with open("./tractions.npy", "rb") as f:
            self.tractions = np.load(f)

    def predict(self, *args) -> np.ndarray:
        return self._predict(self._format_variables(*args))

    def _format_variables(self, n, u, v) -> np.ndarray:
        return n, np.append(u, v)

    def _predict(self, x) -> np.ndarray:
        n = x[0]
        return self.tractions[n]


def tct_elastic_reduced_fixed_displacement():
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
        top_boundary_dofs[2*i:2*i+2] = [node * 2, node * 2 + 1]

    # Bottom BC: Sinusoidal displacement (Time-dependent)
    amplitude = 5.0
    frequency = 1000 # Hz, approx one cycle in 0.003s
    omega = 2 * np.pi * frequency

    # Define displacement function for bottom boundary - will be updated in time loop
    def bottom_displacement_function(t):
        value = amplitude * np.sin(omega * t)
        return Constant(mesh, np.array([0, value]))

    with open("displacement.npy", "rb") as f:
        top_displacement = np.load(f, )

    u_top = Function(V)

    def top_displacement_function(n):
        return top_displacement[n]
    # predictor = Predictor(omega, len(top_boundary_nodes) * mesh.geometry.dim)


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

    # Initialize acceleration to zero
    a_prev.x.array[:] = 0.0  # Important initialization

    beta = 0.25  # Newmark-beta parameter
    gamma = 0.5   # Newmark-beta parameter

    # Temporary Functions for predictor step
    u_pred = Function(V)  # Now a Function object
    v_pred = Function(V)  # Now a Function object

    u_full = []
    v_full = []
    for step in progressbar(range(num_steps)):
        time += dt

        bc_bottom = dirichletbc(bottom_displacement_function(time), bottom_boundary_nodes, V)

        u_top.x.array[top_boundary_dofs] = top_displacement_function(step)
        bc_top = dirichletbc(u_top, top_boundary_nodes)

        # Update current boundary conditions
        current_bcs = [bc_bottom, bc_top]

        # Solve using Newmark-beta
        # Predictor step (using Function objects)
        u_pred.x.array[:] = u_prev.x.array[:] + dt * v_prev.x.array[:] + 0.5 * dt**2 * (1-2*beta) * a_prev.x.array[:]
        v_pred.x.array[:] = v_prev.x.array[:] + dt * (1 - gamma) * a_prev.x.array[:]

        # Solve for acceleration (using u_pred as initial guess)
        u_k.x.array[:] = u_pred.x.array[:]  # Set initial guess for the solver
        problem = LinearProblem(a, L, bcs=current_bcs, u=u_k)
        u_k = problem.solve()

        # Corrector step
        a_k.x.array[:] = (1/(beta*dt**2)) * (u_k.x.array[:] - u_pred.x.array[:]) - ((1-2*beta)/(2*beta)) * a_prev.x.array[:]
        v_k.x.array[:] = v_pred.x.array[:] + dt*gamma*a_k.x.array[:] + dt*(1-gamma)*a_prev.x.array[:]


        if step % 100 == 0:
            u_full.append(u_k.x.array.copy())
            v_full.append(v_k.x.array.copy())

        # Update previous values
        u_prev.x.array[:] = u_k.x.array[:]
        v_prev.x.array[:] = v_k.x.array[:]
        a_prev.x.array[:] = a_k.x.array[:]

    print("Simulation complete")
    return vtk_mesh(mesh), u_full, v_full


if __name__ == "__main__":
    vtk_mesh_obj, u_full_data, v_full_data, interface_nodes = tct_elastic_reduced_fixed_displacement()
