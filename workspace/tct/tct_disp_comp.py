import numpy as np

from tct.tct_disp import TCTExtractDisp
from shared.tct import get_TCT_class_tractions


DEFORMATION = "elastic"  # or plastic


class TCTApplyFixedDisp(get_TCT_class_tractions(DEFORMATION)):

    def __init__(self, disps, frequency: int = 1000) -> None:
        self.disps = disps
        super().__init__(frequency)

    @property
    def height(self) -> float:
        return 25.0

    def _preprocess(self) -> None:
        super()._preprocess()

        self.interface_marker = 88
        self.add_dirichlet_bc(self.interface_boundary, self.interface_marker)

    def _solve_time_step(self) -> None:
        self.update_dirichlet_bc(self.disps[self.step], self.interface_marker)

        self.solve_u()


def compare_disp_application() -> None:
    tct_extract = TCTExtractDisp()
    tct_extract.run()
    tct_extract.postprocess("u", "u", "y", "disps_gen")

    tct_apply = TCTApplyFixedDisp(tct_extract.data_out)
    tct_apply.run()

    tct_apply.postprocess("u", "u", "y", "disps_app")

    bottom_half_nodes_extract = tct_extract.get_nodes(lambda x: x[1] < 25.4, sort=True)
    bottom_half_nodes_apply = tct_apply.get_nodes(lambda x: x[1] < 25.4, sort=True)

    error = np.zeros(tct_extract.formatted_plot_results["u"].shape)
    error[:, bottom_half_nodes_extract] = tct_extract.formatted_plot_results["u"][:, bottom_half_nodes_extract] - tct_apply.formatted_plot_results["u"][:, bottom_half_nodes_apply]

    tct_extract.postprocess(error, "u", "norm", "disps_error")


if __name__ == "__main__":
    compare_disp_application()
