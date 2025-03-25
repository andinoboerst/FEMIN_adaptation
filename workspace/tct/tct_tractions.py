import numpy as np
# import pickle

from shared.tct import TCTSimulation


DEFORMATION = "plastic"  # or plastic


class TCTExtractTractions(TCTSimulation):

    def _preprocess(self) -> None:
        super()._preprocess()

        self.bottom_half_nodes = self.get_nodes(self.bottom_half_p)

        self.data_in = np.zeros((self.num_steps, len(self.interface_dofs)))
        self.data_out = np.zeros((self.num_steps, len(self.interface_dofs)))

    def _solve_time_step(self):
        self.data_in[self.step, :] = self.u_next.x.array[self.interface_dofs]

        self.solve_u()

        self.data_out[self.step, :] = self.calculate_interface_tractions()


class TCTApplyTractions(TCTSimulation):

    height = 25.0

    def __init__(self, predictor, *args, **kwargs) -> None:
        self.predictor = predictor
        super().__init__(*args, **kwargs)

    def _preprocess(self) -> None:
        super()._preprocess()

        self.neumann_interface_marker = 88

        self.add_neumann_bc(self.interface_boundary, self.neumann_interface_marker)

    def _solve_time_step(self) -> None:

        print("To predict: ", self.u_k.x.array[self.interface_dofs])

        prediction = self.predictor.predict(self.u_k.x.array[self.interface_dofs])

        self.update_neumann_bc(prediction, self.neumann_interface_marker)

        self.solve_u()


if __name__ == "__main__":
    tct = TCTExtractTractions(constitutive_model=DEFORMATION)
    # tct.time_total = 5e-4
    # tct.dt = 2e-7
    tct.run()
    tct.postprocess("u", "u", "y", "test5")

    with open("tractions_test5.npy", "wb") as f:
        np.save(f, tct.data_out)

    # with open("results/model_v07.pkl", "rb") as f:
    #     predictor = pickle.load(f)

    # tct = TCTForceApply(predictor)
    # tct.time_total = 5e-4
    # tct.run()

    # tct.postprocess("u_y", "u", "test2")
