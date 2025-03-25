import numpy as np

from tct.tct_tractions import TCTExtractTractions
from shared.tct import TCTSimulation


DEFORMATION = "elastic"  # or plastic


class TCTApplyFixedTractions(TCTSimulation):

    height = 25.0
    # corner_point = (0.0, 25.0)

    def __init__(self, tractions, *args, **kwargs) -> None:
        self.tractions = tractions
        super().__init__(*args, **kwargs)

    def _preprocess(self) -> None:
        super()._preprocess()

        self.neumann_interface_marker = 88

        self.add_neumann_bc(self.interface_boundary, self.neumann_interface_marker)

    def _solve_time_step(self) -> None:
        self.update_neumann_bc(self.tractions[self.step], self.neumann_interface_marker)

        self.solve_u()


def compare_force_application() -> None:
    tct = TCTExtractTractions(frequency=1000, constitutive_model=DEFORMATION)
    tct.run()

    tct.postprocess("u", "u", "y", "tractions_full")

    tct_apply = TCTApplyFixedTractions(tct.data_out, frequency=1000, constitutive_model=DEFORMATION)
    tct_apply.run()

    tct_apply.postprocess("u", "u", "y", "tractions_applied")

    u_k_app_error = np.zeros(tct.formatted_plot_results["u"].shape)
    u_k_app_error[:, tct.bottom_half_nodes, :] = tct.formatted_plot_results["u"][:, tct.bottom_half_nodes, :] - tct_apply.formatted_plot_results["u"][:, tct_apply.bottom_half_nodes, :]
    tct.postprocess(u_k_app_error, "u", "norm", "tractions_applied_error")


if __name__ == "__main__":
    compare_force_application()
