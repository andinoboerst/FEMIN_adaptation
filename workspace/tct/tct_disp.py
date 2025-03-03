import numpy as np
import pickle

from shared.tct import get_TCT_class_tractions


DEFORMATION = "elastic"  # or plastic


class TCTExtractDisp(get_TCT_class_tractions(DEFORMATION)):

    def _preprocess(self) -> None:
        super()._preprocess()

        self.data_in = np.zeros((self.num_steps, len(self.interface_dofs)))
        self.data_out = np.zeros((self.num_steps, len(self.interface_dofs)))

    def _solve_time_step(self):
        self.data_in[self.step, :] = self.calculate_interface_tractions()

        self.solve_u()

        self.data_out[self.step, :] = self.u_k.x.array[self.interface_dofs]


class TCTApplyDisp(get_TCT_class_tractions(DEFORMATION)):

    def __init__(self, predictor, frequency: int = 1000) -> None:
        self.predictor = predictor
        super().__init__(frequency)

    @property
    def height(self) -> float:
        return 25.0

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
