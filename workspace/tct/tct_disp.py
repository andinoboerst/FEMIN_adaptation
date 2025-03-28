import numpy as np
import pickle

from shared.tct import TCTSimulation


DEFORMATION = "elastic"  # or plastic


class TCTExtractDisp(TCTSimulation):

    def _preprocess(self) -> None:
        super()._preprocess()

        self.data_in = np.zeros((self.num_steps, len(self.interface_dofs)), dtype="float64")
        self.data_out = np.zeros((self.num_steps, len(self.interface_dofs)), dtype="float64")

    def _solve_time_step(self):
        self.data_in[self.step, :] = - self.calculate_interface_tractions()

        self.solve_u()

        self.data_out[self.step, :] = self.u_next.x.array[self.interface_dofs]


class TCTApplyDisp(TCTSimulation):

    height = 25.0

    def __init__(self, predictor, *args, **kwargs) -> None:
        self.predictor = predictor
        super().__init__(*args, **kwargs)

    def _preprocess(self) -> None:
        super()._preprocess()

        self.interface_marker = 88
        self.add_dirichlet_bc(self.interface_boundary, self.interface_marker)

    def _solve_time_step(self) -> None:

        prediction = self.predictor.predict(self.calculate_interface_tractions())

        self.update_dirichlet_bc(prediction, self.interface_marker)

        self.solve_u()


if __name__ == "__main__":
    # tct = TCTDispExtract()
    # tct.run()
    # tct.postprocess("u_y", "u", "test4")

    with open("results/model_v02.pkl", "rb") as f:
        predictor = pickle.load(f)

    tct = TCTApplyDisp(predictor)
    tct.time_total = 5e-4
    tct.run()

    tct.postprocess("u_y", "u", "test2")
