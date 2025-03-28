import abc
import numpy as np
from mpi4py import MPI
from dolfinx.fem import Constant, Function, functionspace, dirichletbc, locate_dofs_geometrical
from dolfinx.mesh import create_rectangle, CellType, meshtags, locate_entities
from dolfinx.plot import vtk_mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import TestFunction, TrialFunction, Identity, Measure, grad, inner, dot, tr, dx

from misc.plotting import format_vectors_from_flat, create_mesh_animation
from misc.progress_bar import progressbar


class TCTSimulation(metaclass=abc.ABCMeta):

    E = 200.0e3
    nu = 0.3

    # Bottom BC: Sinusoidal displacement (Time-dependent)
    amplitude = 5.0

    # Time Stepping
    time_total = 3e-3
    dt = 5e-7

    # Newmark-beta parameters
    beta = 0.25
    gamma = 0.5

    # Domain and Mesh
    width = 100.0
    element_size_x = 5.0
    element_size_y = 5.0

    def __init__(self, frequency: int = 1000) -> None:
        self.omega = 2 * np.pi * frequency
    
    def plot_variables(self) -> dict:
        return {
            "u": self.u_k.x.array.copy(),
            "v": self.v_k.x.array.copy(),
            "a": self.a_k.x.array.copy(),
        }

    # @abc.abstractmethod
    def _setup(self) -> None:
        pass
    
    # @abc.abstractmethod
    def _solve_time_step(self) -> None:
        self.solve_u()

    @property
    def height(self) -> float:
        return 50.0

    def setup(self) -> None:
        self._dirichlet_bcs_list = []
        self._neumann_bcs_list = []

        self._define_mesh_functionspace()
        self.dim = self.mesh.geometry.dim

        self._init_variables()

        self._setup()

        self._setup_bcs()

        self.plot_results = {key: [] for key in self.plot_variables().keys()}

    def _define_mesh_functionspace(self) -> None:
        self.nx = int(self.width / self.element_size_x)
        self.ny = int(self.height / self.element_size_y)

        self.mesh = create_rectangle(MPI.COMM_WORLD, cell_type=CellType.quadrilateral,
                            points=((0.0, 0.0), (self.width, self.height)), n=(self.nx, self.ny))

        self.V = functionspace(self.mesh, ("CG", 1, (2,)))
        self.W = functionspace(self.mesh, ("CG", 1, (2, 2)))
        self.sigma_projected = Function(self.W)
        self.w = TestFunction(self.W)
        self.tau = TrialFunction(self.W)

    def _init_variables(self) -> None:
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lmbda = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        self.num_steps = int(self.time_total / self.dt)
        self.time = 0.0

        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.f = Constant(self.mesh, np.array([0.0, 0.0]))  # Force term
        self.a = inner(self.sigma(self.u), self.epsilon(self.v)) * dx
        self.L = inner(self.f, self.v) * dx

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

        self.interface_nodes = self.get_boundary_nodes(self.interface_boundary, sort=True)
        self.interface_dofs = self.get_boundary_dofs(self.interface_nodes)

        self.bottom_boundary_marker = 1111
        self.add_dirichlet_bc(self.bottom_boundary, self.bottom_boundary_marker)
    
    def _bottom_displacement_function(self, t):
        value = self.amplitude * np.sin(self.omega * t)
        return [0, value] * (self.nx + 1)

    def _update_prev_values(self) -> None:
        self.u_prev.x.array[:] = self.u_k.x.array[:]
        self.v_prev.x.array[:] = self.v_k.x.array[:]
        self.a_prev.x.array[:] = self.a_k.x.array[:]

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

    def get_boundary_nodes(self, boundary, sort: bool = False) -> np.array:
        boundary_nodes = locate_dofs_geometrical(self.V, boundary)

        if sort:
            boundary_node_x_coords = np.array([self.mesh.geometry.x[node, 0] for node in boundary_nodes])
            boundary_nodes_sorted_indices = np.argsort(boundary_node_x_coords)
            boundary_nodes = boundary_nodes[boundary_nodes_sorted_indices]
        
        return boundary_nodes

    def get_boundary_dofs(self, boundary_nodes) -> np.array:
        boundary_dofs = np.zeros(len(boundary_nodes) * self.dim, dtype=int)
        for i, node in enumerate(boundary_nodes):
            dofs = [node * self.dim, node * self.dim + 1]
            if self.dim == 3:
                dofs.append(node * self.dim + 2)
            boundary_dofs[self.dim * i:self.dim * i + self.dim] = dofs

        return boundary_dofs
    
    def _setup_bcs(self) -> None:
        self._setup_dirichlet_bcs()
        self._setup_neumann_bcs()

    def _setup_dirichlet_bcs(self) -> None:
        self.current_dirichlet_bcs = []

        self.dirichlet_bcs = {}

        for boundary, marker in self._dirichlet_bcs_list:
            nodes = locate_dofs_geometrical(self.V, boundary)
            dofs = self.get_boundary_dofs(nodes)
            func = Function(self.V)
            self.dirichlet_bcs[marker] = (func, dofs)

            self.current_dirichlet_bcs.append(dirichletbc(func, nodes))

    def _setup_neumann_bcs(self) -> None:
        self.neumann_bcs, facet_indices, facet_markers = {}, [], []
        fdim = self.mesh.topology.dim - 1
        for boundary, marker in self._neumann_bcs_list:
            dofs = self.get_boundary_dofs(locate_dofs_geometrical(self.V, boundary))
            self.neumann_bcs[marker] = (Function(self.V), dofs)
            facets = locate_entities(self.mesh, fdim, boundary)
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker))

        if len(self._neumann_bcs_list) > 0:
            facet_indices = np.hstack(facet_indices).astype(np.int32)
            facet_markers = np.hstack(facet_markers).astype(np.int32)
            sorted_facets = np.argsort(facet_indices)
            facet_tag = meshtags(self.mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

            self.ds = Measure("ds", domain=self.mesh, subdomain_data=facet_tag)

            for marker in self.neumann_bcs:
                self.L += dot(self.neumann_bcs[marker][0], self.v) * self.ds(marker)
    
    def add_dirichlet_bc(self, boundary, marker: int) -> None:
        self._dirichlet_bcs_list.append((boundary, marker))

    def add_neumann_bc(self, boundary, marker: int) -> None:
        self._neumann_bcs_list.append((boundary, marker))
    
    def update_dirichlet_bc(self, values, marker: int) -> None:
        dirichlet = self.dirichlet_bcs[marker]
        dirichlet[0].x.array[dirichlet[1]] = values

    def update_neumann_bc(self, values, marker: int) -> None:
        neumann = self.neumann_bcs[marker]
        neumann[0].x.array[neumann[1]] = values

    def solve_u(self) -> None:
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

    def solve_time_step(self) -> None:
        self.time += self.dt

        self.update_dirichlet_bc(self._bottom_displacement_function(self.time), self.bottom_boundary_marker)

        self._solve_time_step()

        if self.step % 100 == 0:
            for key, res in self.plot_variables().items():
                self.plot_results[key].append(res)

        self._update_prev_values()

    def run(self) -> None:
        self.setup()

        for self.step in progressbar(range(self.num_steps)):
            self.solve_time_step()

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
            variables[var] = format_vectors_from_flat(self.plot_results[var], n_dim=self.dim)

        if scalar_coord == "x":
            scalar_num = 0
        elif scalar_coord == "y":
            scalar_num = 1
        elif scalar_coord == "z":
            scalar_num = 2

        plot_mesh = vtk_mesh(self.mesh)
        create_mesh_animation(plot_mesh, variables[scalar_variable][:, :, scalar_num], variables[vector_variable], name=name)


class TCTElastic(TCTSimulation):
    def sigma(self, u):
        return self.lmbda * tr(self.epsilon(u)) * Identity(self.mesh.geometry.dim) + 2 * self.mu * self.epsilon(u)


class TCTPlastic(TCTSimulation):
    def sigma(self, u):
        return self.lmbda * tr(self.epsilon(u)) * Identity(self.mesh.geometry.dim) + 2 * self.mu * self.epsilon(u)


def tct_force_comp(extractor, applicator, predictor) -> None:

    tct_real = extractor()
    tct_real.setup()

    tct_pred = applicator(predictor)
    tct_pred.setup()

    prediction_error = []
    for step in progressbar(range(tct_real.num_steps)):
        tct_real.step = step
        tct_pred.step = step

        tct_real.solve_time_step()
        tct_pred.solve_time_step()

        prediction_error.append(tct_real.u_k.x.array[tct_real.interface_dofs] - tct_pred.u_k.x.array[tct_pred.interface_dofs])

    def bottom_half_real(x):
        return x[1] <= 25.4

    bottom_half_nodes_real = locate_dofs_geometrical(tct_real.V, bottom_half_real)

    return tct_real, tct_pred, prediction_error, bottom_half_nodes_real


if __name__ == "__main__":
    tct = TCTElastic()
    tct.run()