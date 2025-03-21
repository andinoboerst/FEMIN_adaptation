import numpy as np

from tct.tct_tractions import TCTExtractTractions
from shared.tct import get_TCT_class


DEFORMATION = "elastic"  # or plastic


class TCTApplyFixedTractions(get_TCT_class(DEFORMATION)):

    height = 25.0

    def __init__(self, tractions, frequency: int = 1000) -> None:
        self.tractions = tractions
        super().__init__(frequency)

    def _preprocess(self) -> None:
        super()._preprocess()

        self.bottom_half_nodes = self.get_nodes(self.bottom_half_p)

        self.neumann_interface_marker = 88

        self.add_neumann_bc(self.interface_boundary, self.neumann_interface_marker)

    def _solve_time_step(self) -> None:
        self.update_neumann_bc(self.tractions[self.step], self.neumann_interface_marker)

        self.solve_u()


def compare_force_application() -> None:
    tct = TCTExtractTractions(frequency=575)
    tct.run()

    tct.postprocess("u", "u", "y", "tractions_full")

    tct_apply = TCTApplyFixedTractions(tct.data_out, frequency=575)
    tct_apply.run()

    tct_apply.postprocess("u", "u", "y", "tractions_applied")

    u_k_app_error = np.zeros(tct.formatted_plot_results["u"].shape)
    u_k_app_error[:, tct.bottom_half_nodes, :] = tct.formatted_plot_results["u"][:, tct.bottom_half_nodes, :] - tct_apply.formatted_plot_results["u"][:, tct_apply.bottom_half_nodes, :]
    tct.postprocess(u_k_app_error, "u", "norm", "tractions_applied_error")


def compare_force_application_v2() -> None:
    with open("results/training_out_v11.npy", "rb") as f:
        tractions = np.load(f)[8, :, :]

    tct_apply = TCTApplyFixedTractions(tractions, frequency=1100)
    tct_apply.run()

    tct_apply.postprocess("u", "u", "y", "tractions_applied_test")


if __name__ == "__main__":
    compare_force_application()
