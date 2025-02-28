import numpy as np

from tct.tct_force import TCTForceExtract
from shared.tct import get_TCT_class


DEFORMATION = "elastic"  # or plastic


class TCTForceApplyFixed(get_TCT_class(DEFORMATION)):

    def __init__(self, forces, frequency: int = 1000) -> None:
        self.forces = forces
        super().__init__(frequency)

    @property
    def height(self) -> float:
        return 25.0

    def _setup(self) -> None:
        super()._setup()

        self.neumann_interface_marker = 88

        self.add_neumann_bc(self.interface_boundary, self.neumann_interface_marker)

    def _solve_time_step(self) -> None:
        self.update_neumann_bc(self.forces[self.step], self.neumann_interface_marker)

        self.solve_u()


def compare_force_application() -> None:
    tct_extract = TCTForceExtract()
    tct_extract.run()

    tct_extract.postprocess("u", "u", "y", "forces_gen")

    tct_apply = TCTForceApplyFixed(tct_extract.data_out)
    tct_apply.run()

    tct_apply.postprocess("u", "u", "y", "forces_app")

    bottom_half_nodes_extract = tct_extract.get_nodes(lambda x: x[1] < 25.4, sort=True)
    bottom_half_nodes_apply = tct_apply.get_nodes(lambda x: x[1] < 25.4, sort=True)

    error = np.zeros(tct_extract.formatted_plot_results["u"].shape)
    error[:, bottom_half_nodes_extract] = tct_extract.formatted_plot_results["u"][:, bottom_half_nodes_extract] - tct_apply.formatted_plot_results["u"][:, bottom_half_nodes_apply]

    tct_extract.postprocess(error, "u", "norm", "forces_error")


if __name__ == "__main__":
    compare_force_application()
