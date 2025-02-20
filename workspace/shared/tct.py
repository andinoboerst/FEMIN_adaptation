import abc
import numpy as np
from mpi4py import MPI
from dolfinx.fem import Constant, Function, functionspace, dirichletbc, locate_dofs_geometrical, assemble_scalar, form
from dolfinx.mesh import create_rectangle, CellType, meshtags, locate_entities
from dolfinx.plot import vtk_mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import TestFunction, TrialFunction, Identity, Measure, grad, inner, dot, tr, dx, FacetNormal

from misc.plotting import format_vectors_from_flat, create_mesh_animation
from misc.progress_bar import progressbar


class TCTSimulation(metaclass=abc.ABCMeta):

    E = 200.0e3
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # Bottom BC: Sinusoidal displacement (Time-dependent)
    amplitude = 5.0

    # Time Stepping
    time_total = 3e-3
    dt = 5e-7
    num_steps = int(time_total / dt)
    start_time = 0.0

    # Newmark-beta parameters
    beta = 0.25
    gamma = 0.5

    # Domain and Mesh
    width = 100.0
    element_size_x = 5.0
    element_size_y = 5.0
    nx = int(width / element_size_x)

    def __init__(self, height: float = 50.0, frequency: int = 1000) -> None:
        self.height = height
        self.ny = int(self.height / self.element_size_y)
        self.omega = 2 * np.pi * frequency

        self._define_mesh_functionspace()
        self._init_variables()

        self.setup()
    
    def plot_variables(self) -> tuple:
        return (self.u_k, self.v_k, self.a_k)

    @abc.abstractmethod
    def setup(self) -> None:
        raise NotImplementedError("Must implement setup()")
    
    @abc.abstractmethod
    def solve_time_step(self) -> None:
        raise NotImplementedError("Must implement solve_time_step()")

    def _define_mesh_functionspace(self) -> None:
        self.mesh = create_rectangle(MPI.COMM_WORLD, cell_type=CellType.quadrilateral,
                            points=((0.0, 0.0), (self.width, self.height)), n=(self.nx, self.ny))

        self.V = functionspace(self.mesh, ("CG", 1, (2,)))
        self.W = functionspace(self.mesh, ("CG", 1, (2, 2)))  # Or ("CG", 1) if you want continuous stress
        self.sigma_projected = Function(self.W)
        self.w = TestFunction(self.W)
        self.tau = TrialFunction(self.W)

    def _init_variables(self) -> None:
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.f = Constant(self.mesh, np.array([0.0, 0.0]))  # Force term
        self.a = inner(self.sigma(self.u), self.epsilon(self.v)) * dx
        self.L_body = inner(self.f, self.v) * dx

        self.u_k = Function(self.V, name="Displacement")
        self.u_prev = Function(self.V)
        self.v_k = Function(self.V, name="Velocity")
        self.v_prev = Function(self.V)
        self.a_k = Function(self.V, name="Acceleration")
        self.a_prev = Function(self.V)

        # Initialize displacement and acceleration to zero
        self.u_k.x.array[:] = 0.0
        self.a_prev.x.array[:] = 0.0  # Important initialization

        # Temporary Functions for predictor step
        self.u_pred = Function(self.V)
        self.v_pred = Function(self.V)

        self.dirichlet_bcs = []

    def setup_neumann_bcs(self, bcs: list[tuple]) -> None:
        facet_indices, facet_markers = [], []
        fdim = self.mesh.topology.dim - 1
        for marker, boundary in bcs:
            # marker = 88
            facets = locate_entities(self.mesh, fdim, boundary)
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker))

        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = meshtags(self.mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

        self.ds = Measure("ds", domain=self.mesh, subdomain_data=facet_tag)

    def bottom_displacement_function(self, t):
        value = self.amplitude * np.sin(self.omega * t)
        return Constant(self.mesh, np.array([0, value]))

    @staticmethod
    def top_boundary(x):
        return np.isclose(x[1], 50.0)

    @staticmethod
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)

    @staticmethod
    def interface_boundary(x):
        return np.isclose(x[1], 25.0)
    
    @staticmethod
    def epsilon(u):
        return 0.5 * (grad(u) + grad(u).T)

    def sigma(self, u):
        return self.lmbda * tr(self.epsilon(u)) * Identity(self.mesh.geometry.dim) + 2 * self.mu * self.epsilon(u)

    def boundary_nodes_dofs(self, boundary, sort: bool = False) -> tuple[np.array]:
        boundary_nodes = locate_dofs_geometrical(self.V, boundary)

        if sort:
            boundary_node_x_coords = np.array([self.mesh.geometry.x[node, 0] for node in boundary_nodes])
            boundary_nodes_sorted_indices = np.argsort(boundary_node_x_coords)
            boundary_nodes = boundary_nodes[boundary_nodes_sorted_indices]

        boundary_dofs = np.zeros(len(boundary_nodes) * 2, dtype=int)
        for i, node in enumerate(boundary_nodes):
            boundary_dofs[2 * i:2 * i + 2] = [node * 2, node * 2 + 1]

        boundary_nodes, boundary_dofs

    def solve_u_time_step(self) -> None:
        # Solve using Newmark-beta
        # Predictor step
        self.u_pred.x.array[:] = self.u_prev.x.array[:] + self.dt * self.v_prev.x.array[:] + 0.5 * self.dt**2 * (1 - 2 * self.beta) * self.a_prev.x.array[:]
        self.v_pred.x.array[:] = self.v_prev.x.array[:] + self.dt * (1 - self.gamma) * self.a_prev.x.array[:]

        # Solve for acceleration (using u_pred as initial guess)
        self.u_k.x.array[:] = self.u_pred.x.array[:]  # Set initial guess for the solver
        problem = LinearProblem(self.a, self.L, bcs=self.current_dirichlet_bcs, u=self.u_k)
        self.u_k = problem.solve()

        # Corrector step
        self.a_k.x.array[:] = (1 / (self.beta * self.dt**2)) * (self.u_k.x.array[:] - self.u_pred.x.array[:]) - ((1 - 2 * self.beta) / (2 * self.beta)) * self.a_prev.x.array[:]
        self.v_k.x.array[:] = self.v_pred.x.array[:] + self.dt * self.gamma * self.a_k.x.array[:] + self.dt * (1 - self.gamma) * self.a_prev.x.array[:]
    
    def _update_prev_values(self) -> None:
        self.u_prev.x.array[:] = self.u_k.x.array[:]
        self.v_prev.x.array[:] = self.v_k.x.array[:]
        self.a_prev.x.array[:] = self.a_k.x.array[:]

    def calculate_forces(self, nodes) -> np.array:
        # Project the stress to the function space W
        sigma_expr = self.sigma(self.u_k)

        # Define the *bilinear* and *linear* forms for the projection
        a_proj = inner(self.tau, self.w) * dx  # Bilinear form
        L_proj = inner(sigma_expr, self.w) * dx  # Linear form (same as bilinear in this L2 projection case)

        problem_stress = LinearProblem(a_proj, L_proj, u=self.sigma_projected)  # u=sigma_projected sets sigma_projected as the solution
        problem_stress.solve()

        node_forces = np.zeros(len(nodes) * 2)

        normal_vector = [0, 1]

        for i, node in enumerate(nodes):
            sigma_xx = self.sigma_projected.x.array[4 * node]
            sigma_xy = self.sigma_projected.x.array[4 * node + 1]
            sigma_yy = self.sigma_projected.x.array[4 * node + 3]

            # Compute the force components (accounting for element size)
            node_forces[2 * i] = sigma_xx * normal_vector[0] + sigma_xy * normal_vector[1] #* element_size_y  # Fx
            node_forces[2 * i + 1] = sigma_xy * normal_vector[0] + sigma_yy * normal_vector[1] #* element_size_y  # Fy

        return node_forces
    
    def add_dirichlet_bc(self, bc) -> None:
        self.current_dirichlet_bcs.append(bc)

    def add_neumann_bc(self, values, marker) -> None:
        self.L += dot(values, self.v) * self.ds(marker)

    def run(self) -> None:
        self.plot_results = {[] * len(self.plot_variables())}
        for self.step in progressbar(range(self.num_steps)):
            self.time += self.dt
            self.current_dirichlet_bcs = self.dirichlet_bcs

            self.bc_bottom = dirichletbc(self.bottom_displacement_function(self.time), self.bottom_boundary_nodes, self.V)
            self.add_dirichlet_bc(self.bc_bottom)

            self.L = self.L_body

            self.solve_time_step()

            if self.step % 100 == 0:
                plot_results = self.plot_variables()
                for i, res in enumerate(plot_results):
                    self.plot_results[i].append(res)

            self._update_prev_values()

    def postprocess(self, scalars: str = None, vectors: str = None, name: str = "result") -> None:
        variable_names = []
        if scalars is not None:
            scalar_variable, scalar_coord = scalars.split("_")
            variable_names.append(scalar_variable)
        if vectors is not None:
            vector_variable = vectors
            variable_names.append(vector_variable)

        variable_names = set(variable_names)

        variables = {}
        for var in variable_names:
            if var == "u":
                value = format_vectors_from_flat(self.plot_results[0])
            elif var == "v":
                value = format_vectors_from_flat(self.plot_results[1])
            elif var == "a":
                value = format_vectors_from_flat(self.plot_results[2])
            variables[var] = value

        if scalar_coord == "x":
            scalar_num = 0
        elif scalar_coord == "y":
            scalar_num = 1
        elif scalar_coord == "z":
            scalar_num = 2

        vtk_mesh = vtk_mesh(self.mesh)
        create_mesh_animation(vtk_mesh, variables[scalar_variable][:, :, scalar_num], variables[vector_variable], name=name)
