import numpy as np
from typing import Literal

from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType

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

    def _define_mesh(self) -> None:
        self.nx = int(self.width / self.element_size_x)
        self.ny = int(self.height / self.element_size_y)

        self.mesh = create_rectangle(
            MPI.COMM_WORLD,
            cell_type=CellType.quadrilateral,
            points=((0.0, 0.0), (self.width, self.height)),
            n=(self.nx, self.ny)
        )

    def _init_variables(self) -> None:
        super()._init_variables()

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

    def solve_time_step(self):
        self.update_dirichlet_bc(self._bottom_displacement_function(self.time), self.bottom_boundary_marker)

        return super().solve_time_step()


class TCTElastic(_TCTSimulation, StructuralElasticSimulation):
    pass


class TCTPlastic(_TCTSimulation, StructuralPlasticSimulation):
    pass


def get_TCT_class(deformation: Literal["elastic", "plastic"] = "elastic"):
    deformation = deformation.lower()
    if deformation == "elastic":
        return TCTElastic
    elif deformation == "plastic":
        return TCTPlastic
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
