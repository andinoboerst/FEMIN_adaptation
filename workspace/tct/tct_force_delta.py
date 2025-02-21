import numpy as np
import pickle

from shared.tct import get_TCT_class


DEFORMATION = "elastic"  # or plastic


class TCTForceDeltaExtract(get_TCT_class(DEFORMATION)):

    def _setup(self) -> None:
        self.add_dirichlet_bc(self.top_boundary, 2222)

        self.forces_prev = np.zeros(len(self.interface_dofs))

        self.data_in = np.zeros((self.num_steps, len(self.interface_dofs)))
        self.data_out = np.zeros((self.num_steps, len(self.interface_dofs)))

    def _solve_time_step(self):
        self.data_in[self.step, :] = self.u_k.x.array[self.interface_dofs] - self.u_prev.x.array[self.interface_dofs]

        self.solve_u()

        current_forces = self.calculate_forces(self.interface_nodes)
        self.data_out[self.step, :] = current_forces - self.forces_prev
        self.forces_prev = current_forces


class TCTForceDeltaApply(get_TCT_class(DEFORMATION)):

    def __init__(self, predictor, frequency: int = 1000) -> None:
        self.predictor = predictor
        super().__init__(frequency)

    @property
    def height(self) -> float:
        return 25.0

    def _setup(self) -> None:
        self.neumann_interface_marker = 88

        self.add_neumann_bc(self.interface_boundary, self.neumann_interface_marker)

    def _solve_time_step(self) -> None:
        current_forces = self.calculate_forces(self.interface_nodes)
        predicted_force_delta = self.predictor.predict([self.u_k.x.array[self.interface_dofs] - self.u_prev.x.array[self.interface_dofs]])[0]
        new_forces = current_forces + predicted_force_delta
        self.update_neumann_bc(new_forces, self.neumann_interface_marker)

        self.solve_u()


if __name__ == "__main__":
    tct = TCTForceDeltaExtract()
    tct.run()
    tct.postprocess("u_y", "u", "test5")

    # with open("results/model_v07.pkl", "rb") as f:
    #     predictor = pickle.load(f)

    # tct = TCTForceDeltaApply(predictor)
    # tct.time_total = 5e-4
    # tct.run()

    # tct.postprocess("u_y", "u", "test2")
