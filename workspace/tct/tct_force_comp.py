from tct.tct_force import TCTForceExtract
from shared.tct import get_TCT_class


DEFORMATION = "elastic"  # or plastic


class TCTForceApplyFixed(get_TCT_class(DEFORMATION)):

    def __init__(self, forces, frequency: int = 1000) -> None:
        self.pforces = forces
        super().__init__(frequency)

    @property
    def height(self) -> float:
        return 25.0

    def _setup(self) -> None:
        self.neumann_interface_marker = 88

        self.add_neumann_bc(self.interface_boundary, self.neumann_interface_marker)

    def _solve_time_step(self) -> None:
        self.update_neumann_bc(self.forces[self.step], self.neumann_interface_marker)

        self.solve_u()


def compare_force_application() -> None:
    tct_extract = TCTForceExtract()
    tct_extract.run()
    
    bottom_half_nodes = tct_extract.get_boundary_nodes(lambda x: x[1] < 25.4)
    bottom_half_dofs = tct_extract.get_boundary_dofs(bottom_half_nodes)

    tct_apply = TCTForceApplyFixed(tct_extract.fdata_out)
    tct_apply.run()
