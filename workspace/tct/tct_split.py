import numpy as np
# import pickle

from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType, meshtags, locate_entities
from ufl import TrialFunction, inner, Measure, dx
from dolfinx.fem import Function, Constant, dirichletbc
from dolfinx.fem.petsc import LinearProblem

from shared.tct import get_TCT_class, get_TCT_class_tractions
from shared.progress_bar import progressbar

from tct.tct_disp import TCTExtractDisp
from tct.tct_tractions import TCTExtractTractions


DEFORMATION = "elastic"  # or plastic


class TCTSplitBottom(get_TCT_class(DEFORMATION)):

    height = 25.0

    def _preprocess(self) -> None:
        super()._preprocess()

        self.bottom_half_nodes = self.get_nodes(self.bottom_half)

        self.interface_marker = 55

        self.add_neumann_bc(self.interface_boundary, self.interface_marker)

        self.neumann_data = None

        self.disps = self.u_k.x.array[self.interface_dofs].copy()

    def _solve_time_step(self):
        self.update_neumann_bc(self.neumann_data, self.interface_marker)

        self.solve_u()

        self.disps = self.u_k.x.array[self.interface_dofs].copy()


class TCTSplitTop(get_TCT_class(DEFORMATION)):

    height_t = 50.0
    corner_point = (0.0, 25.0)

    def _init_variables(self):
        super()._init_variables()

        self.f_interface = TrialFunction(self.V)
        self.f_t = Constant(self.mesh, np.array([0.0, 0.0]))
        self.f_res = Function(self.V)

    def _define_mesh(self) -> None:
        self.nx = int(self.width / self.element_size_x)
        self.ny = int(self.height / (self.element_size_y * 2))

        self.mesh = create_rectangle(
            MPI.COMM_WORLD,
            cell_type=CellType.quadrilateral,
            points=(self.corner_point, (self.width, self.height)),
            n=(self.nx, self.ny)
        )

    def _preprocess(self) -> None:
        super()._preprocess()

        self.interface_marker = 33

        self.add_dirichlet_bc(self.interface_boundary, self.interface_marker)

        self.dirichlet_data = None

        self.not_interface_nodes = self.get_nodes(lambda x: x[1] > 25.4)

        facet_indices, facet_markers = [], []

        fdim = self.mesh.topology.dim - 1

        facets = locate_entities(self.mesh, fdim, self.interface_boundary)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, self.interface_marker))

        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = meshtags(self.mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

        self.ds = Measure("ds", domain=self.mesh, subdomain_data=facet_tag)

    def setup(self) -> None:
        super().setup()

        self.tractions = self.calculate_interface_tractions()

    def _define_differential_equations(self):
        super()._define_differential_equations()

        self.a_t = inner(self.f_interface, self.v) * self.ds(self.interface_marker) + inner(self.f_interface, self.v) * dx - inner(self.f_interface, self.v) * dx
        self.L_t = inner(self.sigma(self.u_k), self.epsilon(self.v)) * dx

    def calculate_interface_tractions(self) -> None:

        problem = LinearProblem(
            self.a_t,
            self.L_t,
            bcs=[dirichletbc(Constant(self.mesh, (0.0, 0.0)), self.not_interface_nodes, self.V)],
            u=self.f_res,
            petsc_options=self.linear_petsc_options,
        )
        self.f_res = problem.solve()

        return self.f_res.x.array[self.interface_dofs].copy()

    def _solve_time_step(self):
        self.update_dirichlet_bc(self.dirichlet_data, self.interface_marker)

        self.solve_u()

        self.tractions = self.calculate_interface_tractions()


class TCTSplitBottomV2(get_TCT_class_tractions(DEFORMATION)):

    height = 25.0

    def _preprocess(self) -> None:
        super()._preprocess()

        self.bottom_half_nodes = self.get_nodes(self.bottom_half)

        self.interface_marker = 55

        self.add_dirichlet_bc(self.interface_boundary, self.interface_marker)

        self.dirichlet_data = None

    def setup(self) -> None:
        super().setup()

        self.tractions = self.calculate_interface_tractions()

    def _solve_time_step(self):
        self.update_dirichlet_bc(self.dirichlet_data, self.interface_marker)

        self.solve_u()

        self.tractions = self.calculate_interface_tractions()


class TCTSplitTopV2(get_TCT_class(DEFORMATION)):

    corner_point = (0.0, 25.0)

    def _preprocess(self) -> None:
        super()._preprocess()

        self.interface_marker = 33

        self.add_neumann_bc(self.interface_boundary, self.interface_marker)

        self.neumann_data = None

        self.disps = self.u_k.x.array[self.interface_dofs].copy()

    def _solve_time_step(self):
        self.update_neumann_bc(self.neumann_data, self.interface_marker)

        self.solve_u()

        self.disps = self.u_k.x.array[self.interface_dofs].copy()


def compare_split_tct() -> None:

    # tct_disp = TCTExtractTractions()
    tct_disp = TCTExtractDisp()
    # tct_disp.time_total = 5e-4
    tct_disp.run()
    # disps = tct_disp.data_in
    disps = tct_disp.data_out

    tct_top = TCTSplitTop()
    tct_bottom = TCTSplitBottom()

    # tct_top = TCTSplitTopV2()
    # tct_bottom = TCTSplitBottomV2()

    # tct_top.time_total = 5e-4
    # tct_bottom.time_total = 5e-4

    tct_top.setup()
    tct_bottom.setup()

    for step in progressbar(range(tct_top.num_steps)):
        tct_top.step = step
        tct_bottom.step = step

        tct_top.advance_time()
        tct_bottom.advance_time()

        # Bottom dirichlet, top neumann

        # tct_top.neumann_data = tct_bottom.tractions
        # tct_top.solve_time_step()

        # tct_bottom.dirichlet_data = tct_top.disps
        # tct_bottom.solve_time_step()

        # Bottom neumann, top dirichlet

        tct_top.dirichlet_data = tct_bottom.disps
        # tct_top.dirichlet_data = disps[step]
        print(tct_top.dirichlet_data)
        # print(tct_bottom.disps)
        # print(disps[step])
        tct_top.solve_time_step()

        tct_bottom.neumann_data = tct_top.tractions
        print(tct_bottom.neumann_data)
        tct_bottom.solve_time_step()

        # tct_top.dirichlet_data = tct_bottom.disps
        # tct_top.solve_time_step()

        input("Press Enter to continue...")

    tct_top.format_results()
    tct_bottom.format_results()

    tct_top.postprocess("u", "u", "y", "split_top")
    tct_bottom.postprocess("u", "u", "y", "split_bottom")

    u_k_app_error = np.zeros(tct_disp.formatted_plot_results["u"].shape)
    u_k_app_error[:, tct_disp.bottom_half_nodes, :] = tct_disp.formatted_plot_results["u"][:, tct_disp.bottom_half_nodes, :] - tct_bottom.formatted_plot_results["u"][:, tct_bottom.bottom_half_nodes, :]
    tct_disp.postprocess(u_k_app_error, "u", "norm", "split_error")


if __name__ == "__main__":
    compare_split_tct()
