import numpy as np

from shared.tct import TCTSimulation


class TCTForceExtract(TCTSimulation):

    def _setup(self) -> None:
        self.top_boundary_nodes = self.get_boundary_nodes(self.top_boundary)

        self.add_dirichlet_bc(np.array([0.0, 0.0], dtype=np.float64), self.top_boundary_nodes)

        self.u_interface = np.zeros((self.num_steps, len(self.interface_dofs)))
        self.f_interface = np.zeros((self.num_steps, len(self.interface_dofs)))

    def solve_time_step(self):
        self.u_interface[self.step, :] = self.u_k.x.array[self.interface_dofs]

        self.solve_u()

        self.f_interface[self.step, :] = self.calculate_forces(self.interface_nodes)


class TCTForceApply(TCTSimulation):

    def __init__(self, predictor, frequency: int = 1000) -> None:
        self.predictor = predictor
        super().__init__(frequency)

    def _setup(self) -> None:
        self.neumann_interface_marker = 88

        self.setup_neumann_bcs([(self.neumann_interface_marker, self.interface_boundary, self.interface_dofs)])

    def solve_time_step(self) -> None:
        self.add_neumann_bc(self.predictor.predict([self.u_k.x.array[self.interface_dofs]])[0], self.neumann_interface_marker)

        self.solve_u()


if __name__ == "__main__":
    import pickle
    with open("results/model_v07.pkl", "rb") as f:
        predictor = pickle.load(f)
    tct = TCTForceApply(predictor)
    tct.run()
