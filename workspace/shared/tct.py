import numpy as np
from typing import Literal

from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType, meshtags, locate_entities
from ufl import TrialFunction, TestFunction, inner, Measure
from dolfinx.fem import Function, Constant, functionspace
from dolfinx.fem.petsc import LinearProblem

from shared.fenicsx_sims import FenicsxSimulation, StructuralElasticSimulation, StructuralPlasticSimulation
from shared.progress_bar import progressbar


class _TCTSimulation(FenicsxSimulation):
    # Bottom BC: Sinusoidal displacement (Time-dependent)
    amplitude = 5.0

    # Domain and Mesh
    width = 100.0
    element_size_x = 5.0
    element_size_y = 5.0

    def __init__(self, frequency: int = 1000) -> None:
        super().__init__()

        self.omega = 2 * np.pi * frequency

    @property
    def height(self) -> float:
        return 50.0

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

    def _setup(self) -> None:
        super()._setup()

        self.interface_nodes = self.get_nodes(self.interface_boundary, sort=True)
        self.interface_dofs = self.get_dofs(self.interface_nodes)

        self.bottom_boundary_marker = 1111
        self.add_dirichlet_bc(self.bottom_boundary, self.bottom_boundary_marker)

    def _bottom_displacement_function(self, t):
        value = self.amplitude * np.sin(self.omega * t)
        return [0, value] * (self.nx + 1)

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

    def solve_time_step(self) -> None:
        self.update_dirichlet_bc(self._bottom_displacement_function(self.time), self.bottom_boundary_marker)

        super().solve_time_step()


class _TCTSimulationTractions(_TCTSimulation):

    @property
    def height_t(self) -> float:
        return 25.0

    def _define_mesh(self) -> None:
        super()._define_mesh()

        self.ny_t, self.mesh_t = self.return_mesh(self.height_t)

    def _define_functionspace(self):
        super()._define_functionspace()

        self.V_t = functionspace(self.mesh_t, ("CG", 1, (2,)))

    def _init_variables(self):
        super()._init_variables()

        self.v_t = TestFunction(self.V_t)
        self.f_interface = TrialFunction(self.V_t)
        self.f_f = Constant(self.mesh_t, np.array([0.0, 0.0]))
        self.u_f = Function(self.V_t)
        self.f_res = Function(self.V_t)
        self.dx_t = Measure("dx", domain=self.mesh_t)

    def _setup(self) -> None:
        super()._setup()

        self.add_dirichlet_bc(self.top_boundary, 2222)

        self.interface_nodes_t = self.get_nodes(self.interface_boundary, sort=True, V=self.V_t)
        self.interface_dofs_t = self.get_dofs(self.interface_nodes_t)

        self.bottom_boundary_marker_t = 5555
        self.add_dirichlet_bc(self.bottom_boundary, self.bottom_boundary_marker_t, self.V_t)

        self.not_interface_nodes_t = self.get_nodes(self.not_interface_boundary, sort=True, V=self.V_t)

        self.bottom_half_nodes_t = self.get_nodes(self.bottom_half, sort=True, V=self.V_t)
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
        self.L_t = inner(self.sigma(self.u_f), self.epsilon(self.v_t)) * self.dx_t

    def calculate_interface_tractions(self) -> None:
        self.u_f.x.array[self.bottom_half_dofs_t] = self.u_k.x.array[self.bottom_half_dofs].copy()

        problem = LinearProblem(self.a_t, self.L_t, bcs=self.get_dirichlet_bcs(self.V_t), u=self.f_res)
        self.f_res = problem.solve()

        return self.f_res.x.array[self.interface_dofs_t].copy()

    def solve_time_step(self) -> None:
        self.update_dirichlet_bc(self._bottom_displacement_function(self.time), self.bottom_boundary_marker_t)

        super().solve_time_step()


class TCTElastic(_TCTSimulation, StructuralElasticSimulation):
    pass


class TCTPlastic(_TCTSimulation, StructuralPlasticSimulation):
    pass


class TCTElasticTractions(_TCTSimulationTractions, StructuralElasticSimulation):
    pass


class TCTPlasticTractions(_TCTSimulationTractions, StructuralPlasticSimulation):
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


def tct_comp(extractor: StructuralElasticSimulation, applicator: StructuralElasticSimulation, predictor) -> None:

    tct_real = extractor()
    tct_real.time_total = 2e-4
    tct_real.setup()

    tct_pred = applicator(predictor)
    tct_pred.time_total = 2e-4
    tct_pred.setup()

    prediction_error = []
    for step in progressbar(range(tct_real.num_steps)):
        tct_real.step = step
        tct_pred.step = step
        tct_real.advance_time()
        tct_pred.advance_time()

        tct_real.solve_time_step()
        tct_pred.solve_time_step()

        prediction_error.append(tct_real.u_k.x.array[tct_real.interface_dofs] - tct_pred.u_k.x.array[tct_pred.interface_dofs])

    bottom_half_nodes_real = tct_real.get_nodes(lambda x: x[1] < 25.4, sort=True)
    bottom_half_nodes_pred = tct_pred.get_nodes(lambda x: x[1] < 25.4, sort=True)

    tct_real.format_results()
    tct_pred.format_results()

    return tct_real, tct_pred, prediction_error, bottom_half_nodes_real, bottom_half_nodes_pred
