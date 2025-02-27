import numpy as np

from ufl import TrialFunction, inner, dx, Measure, dot
from dolfinx.fem import Function, Constant, dirichletbc
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import meshtags, locate_entities

from shared.tct import get_TCT_class
from shared.progress_bar import progressbar
from tct.tct_force_comp import TCTForceApplyFixed


class TCTSolveDisp(get_TCT_class("elastic")):

    def _setup(self) -> None:
        self.add_dirichlet_bc(self.top_boundary, 2222)

        self.bottom_half_nodes = self.get_nodes(lambda x: x[1] < 25.4, sort=True)
        self.bottom_half_dofs = self.get_dofs(self.bottom_half_nodes)

        self.not_interface_nodes = self.get_nodes(lambda x: x[1] < 24.6, sort=True)
        self.not_interface_dofs = self.get_dofs(self.not_interface_nodes)

        self.data_in = np.zeros((self.num_steps, len(self.interface_dofs)))

    def _solve_time_step(self):
        self.data_in[self.step, :] = self.u_k.x.array[self.interface_dofs]

        self.solve_u()


class TCTSolveForce(get_TCT_class("elastic")):

    @property
    def height(self) -> float:
        return 25.0

    def _plot_variables(self) -> None:
        return {
            "u": self.u_k.x.array.copy(),
        }

    def _init_variables(self) -> None:
        super()._init_variables()

        self.u_k = Function(self.V)
        self.f_tilde = TrialFunction(self.V)
        self.f_res = Function(self.V)

        self.a = inner(self.f_tilde, self.v) * dx
        self.L = inner(self.sigma(self.u_k), self.epsilon(self.v)) * dx - inner(self.f, self.v) * dx

        self.f_res.x.array[:] = 0.0

        self.bottom_half_nodes = self.get_nodes(lambda x: x[1] < 25.4, sort=True)
        self.bottom_half_dofs = self.get_dofs(self.bottom_half_nodes)

    def _setup(self) -> None:
        super()._setup()

        # self.data_out = np.zeros((self.num_steps, len(self.interface_dofs)))
        self.data_out = np.zeros((self.num_steps, len(self.f_res.x.array)))

    def _solve_time_step(self) -> None:
        self.u_k.x.array[self.bottom_half_dofs] = self.u_set

        problem = LinearProblem(self.a, self.L, bcs=self.current_dirichlet_bcs, u=self.f_res)
        self.f_res = problem.solve()

        # self.data_out[self.step, :] = self.f_res.x.array[self.interface_dofs]
        self.data_out[self.step, :] = self.f_res.x.array


class TCTSolveForceInterface(get_TCT_class("elastic")):

    @property
    def height(self) -> float:
        return 25.0

    def _plot_variables(self) -> None:
        return {
            "u": self.u_k.x.array.copy(),
        }

    @staticmethod
    def not_interface_boundary(x):
        return ~np.isclose(x[1], 25.0)

    def _init_variables(self) -> None:
        super()._init_variables()

        interface_marker = 88
        non_interface_marker = 99
        markers = [
            (self.interface_boundary, interface_marker),
            (self.not_interface_boundary, non_interface_marker),
        ]

        facet_indices, facet_markers = [], []

        fdim = self.mesh.topology.dim - 1

        for boundary, marker in markers:
            facets = locate_entities(self.mesh, fdim, boundary)
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker))

        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = meshtags(self.mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

        ds = Measure("ds", domain=self.mesh, subdomain_data=facet_tag)

        self.u_k = Function(self.V)
        self.f_tilde = TrialFunction(self.V)
        self.f_res = Function(self.V)

        self.a = inner(self.f_tilde, self.v) * ds(interface_marker)
        self.L = inner(self.sigma(self.u_k), self.epsilon(self.v)) * ds(interface_marker)
        # self.L = inner(self.sigma(self.u_k), self.epsilon(self.v)) * dx - inner(self.f, self.v) * dx

        self.f_res.x.array[:] = 0.0

        self.bottom_half_nodes = self.get_nodes(lambda x: x[1] < 25.4, sort=True)
        self.bottom_half_dofs = self.get_dofs(self.bottom_half_nodes)

    def _setup(self) -> None:
        super()._setup()

        self.data_out = np.zeros((self.num_steps, len(self.interface_dofs)))

    def _solve_time_step(self) -> None:
        self.u_k.x.array[self.bottom_half_dofs] = self.u_set

        problem = LinearProblem(self.a, self.L, bcs=self.current_dirichlet_bcs, u=self.f_res)
        self.f_res = problem.solve()

        self.data_out[self.step, :] = self.f_res.x.array[self.interface_dofs]


class TCTSolveForceTest(get_TCT_class("elastic")):

    @property
    def height(self) -> float:
        return 25.0

    def _plot_variables(self) -> None:
        return {
            "u": self.u_f.x.array.copy(),
        }

    def _init_variables(self) -> None:
        super()._init_variables()

        self.f_interface = TrialFunction(self.V)
        self.f_f = Constant(self.mesh, np.array([0.0, 0.0]))
        self.u_f = Function(self.V)
        self.f_res = Function(self.V)

    @staticmethod
    def not_interface_boundary(x):
        return x[1] < 24.6

    def _setup(self) -> None:
        super()._setup()

        self.not_interface_nodes = self.get_nodes(self.not_interface_boundary, sort=True)

        # self.not_interface_marker = 88
        # self.add_dirichlet_bc(self.not_interface_boundary, self.not_interface_marker)

        self.bottom_half_nodes = self.get_nodes(lambda x: x[1] < 25.4, sort=True)
        self.bottom_half_dofs = self.get_dofs(self.bottom_half_nodes)

        self.interface_marker = 88
        self.not_interface_marker = 99

        facet_indices, facet_markers = [], []

        force_markers = [
            (self.interface_boundary, self.interface_marker),
            (self.not_interface_boundary, self.not_interface_marker),
        ]

        fdim = self.mesh.topology.dim - 1

        for boundary, marker in force_markers:
            facets = locate_entities(self.mesh, fdim, boundary)
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker))

        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = meshtags(self.mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

        self.ds = Measure("ds", domain=self.mesh, subdomain_data=facet_tag)

        self.data_out = np.zeros((self.num_steps, len(self.interface_dofs)))

    def _solve_time_step(self) -> None:
        self.u_f.x.array[self.bottom_half_dofs] = self.u_set

        # self.solve_u()
        a_f = inner(self.f_interface, self.v) * self.ds(self.interface_marker) + inner(self.f_interface, self.v) * dx - inner(self.f_interface, self.v) * dx
        L_f = inner(self.sigma(self.u_f), self.epsilon(self.v)) * dx

        problem = LinearProblem(a_f, L_f, bcs=[dirichletbc(Function(self.V), self.not_interface_nodes)], u=self.f_res)
        # problem = LinearProblem(a_f, L_f, u=self.f_res)
        self.f_res = problem.solve()

        # print(self.f_res.x.array[self.interface_dofs])

        self.data_out[self.step, :] = self.f_res.x.array[self.interface_dofs]


class TCTApplyForce(get_TCT_class("elastic")):

    def __init__(self, forces, frequency: int = 1000) -> None:
        self.forces = forces
        super().__init__(frequency)

    @property
    def height(self) -> float:
        return 25.0

    def _init_variables(self) -> None:
        super()._init_variables()

        self.f_res = Function(self.V)

        self.L += inner(self.f_res, self.v) * dx

    def _setup(self) -> None:

        self.bottom_half_nodes = self.get_nodes(lambda x: x[1] < 25.4, sort=True)
        self.bottom_half_dofs = self.get_dofs(self.bottom_half_nodes)

    def _solve_time_step(self):
        self.f_res.x.array[:] = self.forces[self.step]

        self.solve_u()


class TCTApplyForceTest(get_TCT_class("elastic")):

    def __init__(self, forces, frequency: int = 1000) -> None:
        self.forces = forces
        super().__init__(frequency)

    @property
    def height(self) -> float:
        return 25.0

    def _init_variables(self) -> None:
        super()._init_variables()

        self.f_res = Function(self.V)

        self.L += inner(self.f_res, self.v) * dx

    def _setup(self) -> None:

        self.bottom_half_nodes = self.get_nodes(lambda x: x[1] < 25.4, sort=True)
        self.bottom_half_dofs = self.get_dofs(self.bottom_half_nodes)

    def _solve_time_step(self):
        self.f_res.x.array[self.interface_dofs] = self.forces[self.step]

        self.solve_u()


def tct_extraction() -> None:
    tct_d = TCTSolveDisp()
    tct_f = TCTSolveForceTest()

    tct_d.setup()
    tct_f.setup()

    for step in progressbar(range(tct_d.num_steps)):
        tct_d.step = step
        tct_f.step = step

        tct_d.advance_time()

        tct_d.solve_time_step()

        tct_f.u_set = tct_d.u_k.x.array[tct_d.bottom_half_dofs].copy()
        tct_f.solve_time_step()

    tct_d.format_results()
    tct_f.format_results()

    return tct_d, tct_f


if __name__ == "__main__":
    tct_d, tct_f = tct_extraction()

    tct_f.postprocess("u", "u", "y", "forces_half")

    u_k_error = np.zeros(tct_d.formatted_plot_results["u"].shape)
    u_k_error[:, tct_d.bottom_half_nodes, :] = tct_d.formatted_plot_results["u"][:, tct_d.bottom_half_nodes, :] - tct_f.formatted_plot_results["u"][:, tct_f.bottom_half_nodes, :]
    tct_d.postprocess(u_k_error, "u", "norm", "forces_error_test")

    tct_apply = TCTForceApplyFixed(tct_f.data_out)
    tct_apply.run()

    tct_apply.bottom_half_nodes = tct_apply.get_nodes(lambda x: x[1] < 25.4, sort=True)

    tct_apply.postprocess("u", "u", "y", "forces_app")

    u_k_app_error = np.zeros(tct_d.formatted_plot_results["u"].shape)
    u_k_app_error[:, tct_d.bottom_half_nodes, :] = tct_d.formatted_plot_results["u"][:, tct_d.bottom_half_nodes, :] - tct_apply.formatted_plot_results["u"][:, tct_apply.bottom_half_nodes, :]
    tct_d.postprocess(u_k_app_error, "u", "norm", "forces_error_app")
