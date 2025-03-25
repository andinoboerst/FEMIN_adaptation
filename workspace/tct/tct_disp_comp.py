import numpy as np

from tct.tct_disp import TCTExtractDisp
from shared.tct import TCTSimulation
from shared.plotting import format_vectors_from_flat


DEFORMATION = "elastic"  # or plastic


class TCTApplyFixedDisp(TCTSimulation):

    height = 25.0

    def __init__(self, disps, *args, **kwargs) -> None:
        self.disps = disps
        super().__init__(*args, **kwargs)

    def _preprocess(self) -> None:
        super()._preprocess()

        self.interface_marker = 88
        self.add_dirichlet_bc(self.interface_boundary, self.interface_marker)

        self.prediction_input = np.zeros((self.num_steps, len(self.interface_dofs)))

    def _solve_time_step(self) -> None:
        self.prediction_input[self.step, :] = - self.calculate_interface_tractions()

        self.update_dirichlet_bc(self.disps[self.step], self.interface_marker)

        self.solve_u()


def compare_disp_application() -> None:
    tct_extract = TCTExtractDisp(constitutive_model=DEFORMATION)
    tct_extract.run()

    tct_extract.postprocess("u", "u", "y", "disps_full")

    tct_apply = TCTApplyFixedDisp(tct_extract.data_out, constitutive_model=DEFORMATION)
    tct_apply.run()

    tct_apply.postprocess("u", "u", "y", "disps_applied")

    forces_real = tct_extract.data_in
    forces_pred = tct_apply.prediction_input

    with open("forces_real.npy", "wb") as f:
        np.save(f, forces_real)

    with open("forces_pred.npy", "wb") as f:
        np.save(f, forces_pred)

    prediction_error = np.zeros((tct_extract.num_steps, len(tct_extract.u_k.x.array)))
    prediction_error[:, tct_extract.interface_dofs] = np.nan_to_num((forces_real - forces_pred), nan=0.0)
    formatted_prediction_error = format_vectors_from_flat(prediction_error)
    formatted_prediction_error = formatted_prediction_error[::100]

    tct_extract.postprocess(formatted_prediction_error, "u", "norm", "disps_tractions_error")

    error = np.zeros(tct_extract.formatted_plot_results["u"].shape)
    error[:, tct_extract.bottom_half_nodes] = tct_extract.formatted_plot_results["u"][:, tct_extract.bottom_half_nodes] - tct_apply.formatted_plot_results["u"][:, tct_apply.bottom_half_nodes]

    tct_extract.postprocess(error, "u", "norm", "disps_error")


if __name__ == "__main__":
    compare_disp_application()
