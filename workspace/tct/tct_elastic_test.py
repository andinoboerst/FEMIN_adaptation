import numpy as np

from shared.tct import TCTSimulation


class TCTForceExtract(TCTSimulation):

    def _setup(self) -> None:
        self.top_boundary_nodes = self.get_boundary_nodes(self.top_boundary)
        self.interface_nodes = self.get_boundary_nodes(self.interface_boundary, sort=True)
        self.interface_dofs = self.get_boundary_dofs(self.interface_nodes)

        self.add_dirichlet_bc(np.array([0.0, 0.0], dtype=np.float64), self.top_boundary_nodes)

        self.u_interface = np.zeros((self.num_steps, len(self.interface_dofs)))
        self.f_interface = np.zeros((self.num_steps, len(self.interface_dofs)))

    def solve_time_step(self):
        self.u_interface[self.step, :] = self.u_k.x.array[self.interface_dofs]

        self.solve_u()

        self.f_interface[self.step, :] = self.calculate_forces(self.interface_nodes)


if __name__ == "__main__":
    tct = TCTForceExtract()
    tct.run()