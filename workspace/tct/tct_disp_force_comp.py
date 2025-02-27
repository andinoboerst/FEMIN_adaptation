import numpy as np

from tct.tct_disp import TCTDispExtract
from tct.tct_disp_comp import TCTDispApplyFixed

from shared.progress_bar import progressbar
from shared.plotting import format_vectors_from_flat


def tct_force_comp(extractor: TCTDispExtract, applicator: TCTDispApplyFixed) -> None:

    tct_real = extractor()
    # tct_real.time_total = 2e-4
    tct_real.setup()

    tct_pred = applicator([])
    # tct_pred.time_total = 2e-4
    tct_pred.setup()

    tct_pred.data_out = np.zeros((tct_pred.num_steps, len(tct_pred.interface_dofs)))
    for step in progressbar(range(tct_real.num_steps)):
        tct_real.step = step
        tct_pred.step = step
        tct_real.advance_time()
        tct_pred.advance_time()

        tct_real.solve_time_step()
        tct_pred.disps = tct_real.data_out
        tct_pred.data_out[tct_pred.step, :] = tct_pred.calculate_forces(tct_pred.interface_nodes)
        tct_pred.solve_time_step()

    forces_real = tct_real.data_in
    forces_pred = tct_pred.data_out

    prediction_error = np.zeros((tct_real.num_steps, len(tct_real.u_k.x.array)))
    prediction_error[:, tct_real.interface_dofs] = np.nan_to_num((forces_real - forces_pred), nan=0.0)
    formatted_prediction_error = format_vectors_from_flat(prediction_error)

    bottom_half_nodes_real = tct_real.get_nodes(lambda x: x[1] < 25.4, sort=True)
    bottom_half_nodes_pred = tct_pred.get_nodes(lambda x: x[1] < 25.4, sort=True)

    tct_real.format_results()
    tct_pred.format_results()

    return tct_real, tct_pred, formatted_prediction_error, bottom_half_nodes_real, bottom_half_nodes_pred


if __name__ == "__main__":
    tct_real, tct_pred, prediction_error, bottom_half_nodes_real, bottom_half_nodes_pred = tct_force_comp(TCTDispExtract, TCTDispApplyFixed)

    tct_real.postprocess(prediction_error, "u", "norm", "disps_force_error")
