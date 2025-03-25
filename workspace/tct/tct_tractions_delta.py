import numpy as np
# import pickle

from shared.tct import TCTSimulation


DEFORMATION = "elastic"  # or plastic


class TCTExtractTractionsDelta(TCTSimulation):

    def _preprocess(self) -> None:
        super()._preprocess()

        self.tractions_prev = np.zeros(len(self.interface_dofs))

        self.data_in = np.zeros((self.num_steps, len(self.interface_dofs)))
        self.data_out = np.zeros((self.num_steps, len(self.interface_dofs)))

    def _solve_time_step(self):
        self.data_in[self.step, :] = self.u_k.x.array[self.interface_dofs] - self.u_prev.x.array[self.interface_dofs]

        self.solve_u()

        current_tractions = self.calculate_interface_tractions()
        self.data_out[self.step, :] = current_tractions - self.tractions_prev
        self.tractions_prev = current_tractions


class TCTApplyTractionsDelta(TCTSimulation):

    height = 25.0

    def __init__(self, predictor, *args, **kwargs) -> None:
        self.predictor = predictor
        super().__init__(*args, **kwargs)

    def _preprocess(self) -> None:
        super()._preprocess()

        self.neumann_interface_marker = 88

        self.add_neumann_bc(self.interface_boundary, self.neumann_interface_marker)

    def _solve_time_step(self) -> None:
        current_tractions = self.calculate_interface_tractions()
        predicted_tractions_delta = self.predictor.predict(self.u_k.x.array[self.interface_dofs] - self.u_prev.x.array[self.interface_dofs])
        new_tractions = current_tractions + predicted_tractions_delta
        self.update_neumann_bc(new_tractions, self.neumann_interface_marker)

        self.solve_u()


if __name__ == "__main__":
    tct = TCTExtractTractionsDelta(constitutive_model=DEFORMATION)
    tct.run()
    tct.postprocess("u_y", "u", "test5")

    # with open("results/model_v07.pkl", "rb") as f:
    #     predictor = pickle.load(f)

    # tct = TCTApplyTractionsDelta(predictor)
    # tct.time_total = 5e-4
    # tct.run()

    # tct.postprocess("u_y", "u", "test2")
