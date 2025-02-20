from shared.tct import TCTSimulation


class TCTTest(TCTSimulation):

    def setup(self) -> None:
        pass

    def solve_time_step(self):
        self.solve_u_time_step()