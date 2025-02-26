import numpy as np
import pickle

from shared.tct import get_TCT_class


DEFORMATION = "elastic"  # or plastic


class TCTDispDeltaExtract(get_TCT_class(DEFORMATION)):

    def _setup(self) -> None:
        self.add_dirichlet_bc(self.top_boundary, 2222)

        self.forces_prev = np.zeros(len(self.interface_dofs))

        self.data_in = np.zeros((self.num_steps, len(self.interface_dofs)))
        self.data_out = np.zeros((self.num_steps, len(self.interface_dofs)))

    def _solve_time_step(self):
        current_forces = self.calculate_forces(self.interface_nodes)
        self.data_in[self.step, :] = current_forces - self.forces_prev
        self.forces_prev = current_forces

        self.solve_u()

        self.data_out[self.step, :] = self.u_k.x.array[self.interface_dofs] - self.u_prev.x.array[self.interface_dofs]


class TCTDispDeltaApply(get_TCT_class(DEFORMATION)):

    def __init__(self, predictor, frequency: int = 1000) -> None:
        self.predictor = predictor
        super().__init__(frequency)

    @property
    def height(self) -> float:
        return 25.0

    def _setup(self) -> None:
        self.interface_marker = 88
        self.add_dirichlet_bc(self.interface_boundary, self.interface_marker)

        self.forces_prev = np.zeros(len(self.interface_dofs))

    def _solve_time_step(self) -> None:
        current_forces = self.calculate_forces(self.interface_nodes)
        predicted_u_delta = self.predictor.predict(current_forces - self.forces_prev)
        self.forces_prev = current_forces

        u_interface = self.u_k.x.array[self.interface_dofs] + predicted_u_delta

        self.update_dirichlet_bc(u_interface, self.interface_marker)

        self.solve_u()


if __name__ == "__main__":
    # tct = TCTDispDeltaExtract()
    # tct.run()
    # tct.postprocess("u_y", "u", "test4")

    with open("results/model_v02.pkl", "rb") as f:
        predictor = pickle.load(f)

    tct = TCTDispDeltaApply(predictor)
    tct.time_total = 5e-4
    tct.run()

    tct.postprocess("u_y", "u", "test2")
