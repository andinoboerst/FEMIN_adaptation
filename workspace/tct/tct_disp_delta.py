import numpy as np
import pickle

from shared.tct import TCTSimulation


DEFORMATION = "elastic"  # or plastic


class TCTExtractDispDelta(TCTSimulation):

    def _preprocess(self) -> None:
        super()._preprocess()
        self.add_dirichlet_bc(self.top_boundary, 2222)

        self.tractions_prev = np.zeros(len(self.interface_dofs))

        self.data_in = np.zeros((self.num_steps, len(self.interface_dofs)))
        self.data_out = np.zeros((self.num_steps, len(self.interface_dofs)))

    def _solve_time_step(self):
        current_tractions = self.calculate_interface_tractions()
        self.data_in[self.step, :] = current_tractions - self.tractions_prev
        self.tractions_prev = current_tractions

        self.solve_u()

        self.data_out[self.step, :] = self.u_k.x.array[self.interface_dofs] - self.u_prev.x.array[self.interface_dofs]


class TCTApplyDispDelta(TCTSimulation):

    height = 25.0

    def __init__(self, predictor, *args, **kwargs) -> None:
        self.predictor = predictor
        super().__init__(*args, **kwargs)

    def _preprocess(self) -> None:
        super()._preprocess()
        self.interface_marker = 88
        self.add_dirichlet_bc(self.interface_boundary, self.interface_marker)

        self.tractions_prev = np.zeros(len(self.interface_dofs))

    def _solve_time_step(self) -> None:
        current_tractions = self.calculate_interface_tractions()
        predicted_u_delta = self.predictor.predict(current_tractions - self.tractions_prev)
        self.tractions_prev = current_tractions

        u_interface = self.u_k.x.array[self.interface_dofs] + predicted_u_delta

        self.update_dirichlet_bc(u_interface, self.interface_marker)

        self.solve_u()


if __name__ == "__main__":
    # tct = TCTExtractDispDelta()
    # tct.run()
    # tct.postprocess("u_y", "u", "test4")

    with open("results/model_v02.pkl", "rb") as f:
        predictor = pickle.load(f)

    tct = TCTApplyDispDelta(predictor, constitutive_model=DEFORMATION)
    tct.time_total = 5e-4
    tct.run()

    tct.postprocess("u_y", "u", "test2")
