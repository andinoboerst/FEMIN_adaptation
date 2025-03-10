import numpy as np
from typing import Literal

from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType, meshtags, locate_entities
from ufl import TrialFunction, TestFunction, inner, Measure, sqrt, conditional
from dolfinx.fem import Function, Constant, functionspace, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem

from shared.fenicsx_sims import FenicsxSimulation, StructuralElasticSimulation, StructuralPlasticSimulation


class _TCTSimulation(FenicsxSimulation):
    # Bottom BC: Sinusoidal displacement (Time-dependent)
    amplitude = 5.0

    # Domain and Mesh
    width = 100.0
    height = 50.0
    element_size_x = 5.0
    element_size_y = 5.0

    def __init__(self, frequency: int = 1000) -> None:
        super().__init__()

        self.omega = 2 * np.pi * frequency

    def return_mesh(self, height: float):
        ny = int(height / self.element_size_y)

        mesh = create_rectangle(
            MPI.COMM_WORLD,
            cell_type=CellType.quadrilateral,
            points=((0.0, 0.0), (self.width, height)),
            n=(self.nx, ny)
        )

        return ny, mesh

    def _define_mesh(self) -> None:
        self.nx = int(self.width / self.element_size_x)

        self.ny, self.mesh = self.return_mesh(self.height)

    def _preprocess(self) -> None:
        super()._preprocess()

        self.add_dirichlet_bc(self.top_boundary, 2222)

        self.interface_nodes = self.get_nodes(self.interface_boundary)
        self.interface_dofs = self.get_dofs(self.interface_nodes)

        self.bottom_nodes = self.get_nodes(self.bottom_boundary)

        self.bottom_boundary_marker = 1111
        self.add_dirichlet_bc(self.bottom_boundary, self.bottom_boundary_marker)

    def bottom_displacement_function(self, t):
        value = self.amplitude * np.sin(self.omega * t)
        return np.array([0, value] * len(self.bottom_nodes), dtype=float)

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
    def bottom_half(x):
        return x[1] < 25.4

    @staticmethod
    def not_interface_boundary(x):
        return x[1] < 24.6

    # @staticmethod
    # def bottom_half(x):
    #     return x[1] <= 25 * np.isclose(x[1], 25)

    # @staticmethod
    # def not_interface_boundary(x):
    #     return x[1] < 25 * ~np.isclose(x[1], 25)

    def solve_time_step(self) -> None:
        self.update_dirichlet_bc(self.bottom_displacement_function(self.time), self.bottom_boundary_marker)

        super().solve_time_step()


class _TCTSimulationTractions(_TCTSimulation):

    height_t = 25.0

    def _define_mesh(self) -> None:
        super()._define_mesh()

        self.ny_t, self.mesh_t = self.return_mesh(self.height_t)

    def _define_functionspace(self):
        super()._define_functionspace()

        self.V_t = functionspace(self.mesh_t, (self.element_type, 1, (2,)))

    def _init_variables(self):
        super()._init_variables()

        self.v_t = TestFunction(self.V_t)
        self.f_interface = TrialFunction(self.V_t)
        self.f_t = Constant(self.mesh_t, np.array([0.0, 0.0]))
        self.u_t = Function(self.V_t)
        self.f_res = Function(self.V_t)
        self.dx_t = Measure("dx", domain=self.mesh_t)

    def _preprocess(self) -> None:
        super()._preprocess()

        # Full simulation
        self.add_dirichlet_bc(self.top_boundary, 2222)

        self.bottom_half_nodes = self.get_nodes(self.bottom_half)
        self.bottom_half_dofs = self.get_dofs(self.bottom_half_nodes)

        # Traction extraction
        self.interface_nodes_t = self.get_nodes(self.interface_boundary, V=self.V_t)
        self.interface_dofs_t = self.get_dofs(self.interface_nodes_t)

        self.not_interface_nodes_t = self.get_nodes(self.not_interface_boundary, V=self.V_t)

        self.bottom_half_nodes_t = self.get_nodes(self.bottom_half, V=self.V_t)
        self.bottom_half_dofs_t = self.get_dofs(self.bottom_half_nodes_t)

        self.interface_marker_t = 88
        self.not_interface_marker_t = 99

        facet_indices, facet_markers = [], []

        fdim = self.mesh.topology.dim - 1

        facets = locate_entities(self.mesh_t, fdim, self.interface_boundary)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, self.interface_marker_t))

        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = meshtags(self.mesh_t, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

        self.ds_t = Measure("ds", domain=self.mesh_t, subdomain_data=facet_tag)

        self.add_dirichlet_bc(self.not_interface_boundary, 1234, self.V_t)

    def _define_differential_equations(self):
        super()._define_differential_equations()

        self.a_t = inner(self.f_interface, self.v_t) * self.ds_t(self.interface_marker_t) + inner(self.f_interface, self.v_t) * self.dx_t - inner(self.f_interface, self.v_t) * self.dx_t
        self.L_t = inner(self.sigma(self.u_t), self.epsilon(self.v_t)) * self.dx_t

    def calculate_interface_tractions(self) -> None:
        self.u_t.x.array[self.bottom_half_dofs_t] = self.u_k.x.array[self.bottom_half_dofs].copy()

        problem = LinearProblem(self.a_t, self.L_t, bcs=self.get_dirichlet_bcs(self.V_t), u=self.f_res)
        self.f_res = problem.solve()

        return self.f_res.x.array[self.interface_dofs_t].copy()


class _TCTSimulationTractionsPlastic(_TCTSimulation):

    @property
    def height_t(self) -> float:
        return 25.0

    def _define_mesh(self) -> None:
        super()._define_mesh()

        self.ny_t, self.mesh_t = self.return_mesh(self.height_t)

    def _define_functionspace(self):
        super()._define_functionspace()

        self.V_t = functionspace(self.mesh_t, ("CG", 1, (2,)))
        self.W_t = functionspace(self.mesh_t, ("DG", 0, (2, 2)))

    def _init_variables(self):
        super()._init_variables()

        self.v_t = TestFunction(self.V_t)
        self.f_interface = TrialFunction(self.V_t)
        self.f_t = Constant(self.mesh_t, np.array([0.0, 0.0]))
        self.f_res = Function(self.V_t)
        self.dx_t = Measure("dx", domain=self.mesh_t)

        self.sigma_new_t = Function(self.W_t)

    def _preprocess(self) -> None:
        super()._preprocess()

        # Full simulation
        # self.add_dirichlet_bc(self.top_boundary, 2222)

        self.bottom_half_nodes = self.get_nodes(self.bottom_half)
        self.bottom_half_dofs = self.get_dofs(self.bottom_half_nodes)

        # Traction extraction
        self.interface_nodes_t = self.get_nodes(self.interface_boundary, V=self.V_t)
        self.interface_dofs_t = self.get_dofs(self.interface_nodes_t)

        self.bottom_boundary_marker_t = 5555
        self.add_dirichlet_bc(self.bottom_boundary, self.bottom_boundary_marker_t, self.V_t)

        self.not_interface_nodes_t = self.get_nodes(self.not_interface_boundary, V=self.V_t)

        self.bottom_half_nodes_t = self.get_nodes(self.bottom_half, V=self.V_t)
        self.bottom_half_dofs_sigma_t = self.get_dofs(self.bottom_half_nodes_t, value_dim=2)
        self.bottom_half_dofs_sigma = self.get_dofs(self.bottom_half_nodes, value_dim=2)

        self.interface_marker_t = 88
        self.not_interface_marker_t = 99

        facet_indices, facet_markers = [], []

        fdim = self.mesh.topology.dim - 1

        facets = locate_entities(self.mesh_t, fdim, self.interface_boundary)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, self.interface_marker_t))

        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = meshtags(self.mesh_t, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

        self.ds_t = Measure("ds", domain=self.mesh_t, subdomain_data=facet_tag)

        self.add_dirichlet_bc(self.not_interface_boundary, 1234, self.V_t)

    def _define_differential_equations(self):
        super()._define_differential_equations()

        # Define variational problem
        self.residual_t = self.apply_neumann_bcs(inner(self.f_res, self.v_t) * self.ds_t(self.interface_marker_t) + inner(self.f_t, self.v_t) * self.dx_t - inner(self.sigma_new_t, self.epsilon(self.v_t)) * self.dx_t, self.V_t)

        self.a_t = inner(self.f_interface, self.v_t) * self.ds_t(self.interface_marker_t) + inner(self.f_interface, self.v_t) * self.dx_t - inner(self.sigma_new_t, self.epsilon(self.v_t)) * self.dx_t
        self.L_t = inner(self.sigma_new_t, self.epsilon(self.v_t)) * self.dx_t

    def calculate_interface_tractions(self) -> None:
        # self.u_t.x.array[self.bottom_half_dofs_t] = self.u_k.x.array[self.bottom_half_dofs].copy()

        # problem = NonlinearProblem(self.residual_t, self.f_res, bcs=self.get_dirichlet_bcs(self.V_t))
        # solver = NewtonSolver(MPI.COMM_WORLD, problem)
        # solver.solve(self.u_k)

        self.sigma_new_t.x.array[self.bottom_half_dofs_sigma_t] = self.sigma_new.x.array[self.bottom_half_dofs_sigma].copy()

        problem = LinearProblem(self.a_t, self.L_t, bcs=self.get_dirichlet_bcs(self.V_t), u=self.f_res)
        self.f_res = problem.solve()

        return self.f_res.x.array[self.interface_dofs_t].copy()

    def solve_time_step(self) -> None:
        self.update_dirichlet_bc(self.bottom_displacement_function(self.time), self.bottom_boundary_marker_t)

        super().solve_time_step()


class TCTElastic(_TCTSimulation, StructuralElasticSimulation):
    pass


class TCTPlastic(_TCTSimulation, StructuralPlasticSimulation):
    pass


class TCTElasticTractions(_TCTSimulationTractions, StructuralElasticSimulation):
    pass


class TCTPlasticTractions(_TCTSimulationTractionsPlastic, StructuralPlasticSimulation):
    pass


def get_TCT_class(deformation: Literal["elastic", "plastic"] = "elastic"):
    deformation = deformation.lower()
    if deformation == "elastic":
        return TCTElastic
    elif deformation == "plastic":
        return TCTPlastic
    else:
        raise ValueError(f"Unknown simulation deformation: {deformation}.")


def get_TCT_class_tractions(deformation: Literal["elastic", "plastic"] = "elastic"):
    deformation = deformation.lower()
    if deformation == "elastic":
        return TCTElasticTractions
    elif deformation == "plastic":
        return TCTPlasticTractions
    else:
        raise ValueError(f"Unknown simulation deformation: {deformation}.")


def tct_comp(extractor: StructuralElasticSimulation, applicator: StructuralElasticSimulation, filename: str) -> None:

    extractor.run()

    extractor.postprocess("u", "u", "y", f"{filename}_full")

    applicator.run()

    applicator.postprocess("u", "u", "y", f"{filename}_applied")

    applicator.bottom_half_nodes = applicator.get_nodes(lambda x: x[1] < 25.4, sort=True)

    u_k_app_error = np.zeros(extractor.formatted_plot_results["u"].shape)
    u_k_app_error[:, extractor.bottom_half_nodes, :] = extractor.formatted_plot_results["u"][:, extractor.bottom_half_nodes, :] - applicator.formatted_plot_results["u"][:, applicator.bottom_half_nodes, :]
    extractor.postprocess(u_k_app_error, "u", "norm", f"{filename}_applied_error")
