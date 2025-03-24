import numpy as np
from typing import Literal

from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType

from shared.fenicsx_sims import StructuralSimulation


class TCTSimulation(StructuralSimulation):
    # Bottom BC: Sinusoidal displacement (Time-dependent)
    amplitude = 5.0

    # Domain and Mesh
    width = 100.0
    height = 50.0
    element_size_x = 5.0
    element_size_y = 5.0
    corner_point = (0.0, 0.0)

    def __init__(self, frequency: int = 1000, constitutive_model: Literal["elastic", "plastic"] = "elastic") -> None:
        super().__init__()

        self.omega = 2 * np.pi * frequency

        self.constitutive_model = constitutive_model

    def return_mesh(self, height: float, corner_point=None):
        if corner_point is None:
            corner_point = self.corner_point
        nx = int(self.width / self.element_size_x)
        ny = int((height - corner_point[1]) / self.element_size_y)

        mesh = create_rectangle(
            MPI.COMM_WORLD,
            cell_type=CellType.quadrilateral,
            points=(corner_point, (self.width, height)),
            n=(nx, ny)
        )

        return mesh

    def _define_mesh(self) -> None:
        self.mesh = self.return_mesh(self.height)

        self.mesh_t = self.return_mesh(25.0)

    def _preprocess(self) -> None:
        super()._preprocess()

        self.add_dirichlet_bc(self.top_boundary, 2222)

        self.interface_nodes = self.get_nodes(self.interface_boundary)
        self.interface_dofs = self.get_dofs(self.interface_nodes)

        self.bottom_nodes = self.get_nodes(self.bottom_boundary)

        self.bottom_boundary_marker = 1111
        self.add_dirichlet_bc(self.bottom_boundary, self.bottom_boundary_marker)

        self.traction_parameters = self.setup_traction_problem(
            self.mesh_t,
            self.mesh,
            self.bottom_half_p,
            self.bottom_half_e,
            self.interface_boundary,
        )

    def _define_differential_equations(self) -> None:
        super()._define_differential_equations()

        self.traction_problem = self.get_traction_problem(*self.traction_parameters)

    def bottom_displacement_function(self, t):
        value = self.amplitude * np.sin(self.omega * t)
        # value = abs(self.amplitude * np.sin(self.omega * t))
        # value = self.amplitude / 2 * np.cos(self.omega * t) - 2.5
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
    def bottom_half_p(x):
        return x[1] < 25.4

    @staticmethod
    def bottom_half_e(x):
        return x[1] < 25.0

    @staticmethod
    def not_interface_boundary(x):
        return x[1] < 24.6

    def solve_time_step(self) -> None:
        self.update_dirichlet_bc(self.bottom_displacement_function(self.time), self.bottom_boundary_marker)

        super().solve_time_step()


def tct_comp(extractor: StructuralSimulation, applicator: StructuralSimulation, filename: str) -> None:

    extractor.run()

    extractor.postprocess("u", "u", "y", f"{filename}_full")

    applicator.run()

    applicator.postprocess("u", "u", "y", f"{filename}_applied")

    applicator.bottom_half_nodes = applicator.get_nodes(lambda x: x[1] < 25.4, sort=True)

    u_k_app_error = np.zeros(extractor.formatted_plot_results["u"].shape)
    u_k_app_error[:, extractor.bottom_half_nodes, :] = extractor.formatted_plot_results["u"][:, extractor.bottom_half_nodes, :] - applicator.formatted_plot_results["u"][:, applicator.bottom_half_nodes, :]
    extractor.postprocess(u_k_app_error, "u", "norm", f"{filename}_applied_error")
