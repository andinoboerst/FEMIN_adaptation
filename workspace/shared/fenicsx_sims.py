import abc
import numpy as np
import logging
from functools import partial
# import dill

from mpi4py import MPI
from dolfinx.fem import Constant, Function, functionspace, dirichletbc, locate_dofs_geometrical
from dolfinx.mesh import meshtags, locate_entities
from dolfinx.plot import vtk_mesh
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from ufl import TestFunction, TrialFunction, Identity, Measure, grad, inner, dot, tr, sqrt, conditional, sym, gt, eq, And, dx
from petsc4py import PETSc

from shared.plotting import format_vectors_from_flat, create_mesh_animation
from shared.progress_bar import progressbar

logger = logging.getLogger("fenicsx_sims")


class FenicsxSimulation(metaclass=abc.ABCMeta):
    # Time Stepping
    time_total = 3e-3
    dt = 5e-7

    element_type = ("CG", 1)
    element_type_sigma = ("DG", 0)

    linear_petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    def __init__(self) -> None:
        pass

    def _plot_variables(self) -> dict:
        return {}

    @property
    def plot_variables_options(self) -> list:
        try:
            return self.plot_results.keys()
        except AttributeError:
            logger.warning("Simulation has not been set up yet.")
            return []

    # @abc.abstractmethod
    def _preprocess(self) -> None:
        pass

    @abc.abstractmethod
    def _solve_time_step(self) -> None:
        raise NotImplementedError("Need to implement _solve_time_step()")

    @abc.abstractmethod
    def _define_mesh(self) -> None:
        raise NotImplementedError("Need to implement _define_mesh()")

    @abc.abstractmethod
    def _define_functionspace(self) -> None:
        raise NotImplementedError("Need to implement _define_functionspace()")

    @abc.abstractmethod
    def _init_variables(self) -> None:
        raise NotImplementedError("Need to implement _init_variables()")

    @abc.abstractmethod
    def _define_differential_equations(self) -> None:
        raise NotImplementedError("Need to implement _define_differential_equations()")

    # def save(self, path: str) -> None:
    #     with open(path, "wb") as f:
    #         dill.dump(self, f)

    # def load(self, path: str) -> None:
    #     with open(path, "rb") as f:
    #         return dill.load(f)

    def setup(self) -> None:
        self._dirichlet_bcs_list = []
        self._neumann_bcs_list = []

        self.num_steps = int(self.time_total / self.dt)
        self.time = 0.0

        self._define_mesh()
        self.dim = self.mesh.geometry.dim

        self._define_functionspace()

        self._init_variables()

        self._preprocess()

        self._setup_bcs()

        self._define_differential_equations()

        self.plot_results = {key: [] for key in self._plot_variables().keys()}

    def get_nodes(self, marker, V=None) -> np.array:
        if V is None:
            V = self.V

        return locate_dofs_geometrical(V, marker)

    def get_dofs(self, nodes, value_dim: int = 1) -> np.array:
        dofs = np.zeros(len(nodes) * self.dim * value_dim, dtype=int)
        for i, node in enumerate(nodes):
            start = node * self.dim * value_dim
            node_dofs = range(start, start + (self.dim * value_dim))
            dofs[self.dim * value_dim * i:self.dim * value_dim * i + (self.dim * value_dim)] = node_dofs

        return dofs

    def _setup_bcs(self) -> None:
        self._setup_dirichlet_bcs()
        self._setup_neumann_bcs()

    def _setup_dirichlet_bcs(self) -> None:
        self._applied_dirichlet_bcs = ([], [])

        self._dirichlet_bcs = {}

        for boundary, marker, V in self._dirichlet_bcs_list:
            nodes = self.get_nodes(boundary, V=V)
            dofs = self.get_dofs(nodes)
            func = Function(V)
            self._dirichlet_bcs[marker] = (func, dofs)

            if V not in self._applied_dirichlet_bcs[0]:
                self._applied_dirichlet_bcs[0].append(V)
                self._applied_dirichlet_bcs[1].append([])

            i = self._applied_dirichlet_bcs[0].index(V)
            self._applied_dirichlet_bcs[1][i].append(dirichletbc(func, nodes))

    def get_dirichlet_bcs(self, V=None):
        if V is None:
            V = self.V

        try:
            index = self._applied_dirichlet_bcs[0].index(V)
            return self._applied_dirichlet_bcs[1][index]
        except ValueError:
            return []

    def _setup_neumann_bcs(self) -> None:
        self._neumann_bcs, facet_info = {}, {}
        for boundary, marker, V in self._neumann_bcs_list:
            mesh = V.mesh
            fdim = mesh.topology.dim - 1
            dofs = self.get_dofs(locate_dofs_geometrical(V, boundary))
            self._neumann_bcs[marker] = (Function(V), dofs)
            facets = locate_entities(mesh, fdim, boundary)
            if mesh not in facet_info:
                facet_info[mesh] = (
                    [],
                    [],
                    [],
                    V
                )

            facet_info[mesh][0].append(facets)
            facet_info[mesh][1].append(np.full_like(facets, marker))
            facet_info[mesh][2].append(marker)

        self._applied_neumann_bcs = ([], [])
        for mesh, facet_info in facet_info.items():
            dx_ = Measure("dx", domain=mesh)
            v = TestFunction(V)
            L = inner(Constant(mesh, np.array([0.0] * self.dim)), v) * dx_
            facet_indices = np.hstack(facet_info[0]).astype(np.int32)
            facet_markers = np.hstack(facet_info[1]).astype(np.int32)
            sorted_facets = np.argsort(facet_indices)
            facet_tag = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

            ds = Measure("ds", domain=mesh, subdomain_data=facet_tag)

            for marker in facet_info[2]:
                L += dot(self._neumann_bcs[marker][0], v) * ds(marker)

            self._applied_neumann_bcs[0].append(facet_info[3])
            self._applied_neumann_bcs[1].append(L)

    def apply_neumann_bcs(self, L, V=None):
        if V is None:
            V = self.V

        try:
            index = self._applied_neumann_bcs[0].index(V)
            L += self._applied_neumann_bcs[1][index]
        except ValueError:
            logger.warning("No Neumann BCs applied to this function space.")

        return L

    def add_dirichlet_bc(self, boundary, marker: int, V=None) -> None:
        if V is None:
            V = self.V

        self._dirichlet_bcs_list.append((boundary, marker, V))

    def add_neumann_bc(self, boundary, marker: int, V=None) -> None:
        if V is None:
            V = self.V

        self._neumann_bcs_list.append((boundary, marker, V))

    def update_dirichlet_bc(self, values, marker: int) -> None:
        dirichlet = self._dirichlet_bcs[marker]
        dirichlet[0].x.array[dirichlet[1]] = values

    def update_neumann_bc(self, values, marker: int) -> None:
        neumann = self._neumann_bcs[marker]
        neumann[0].x.array[neumann[1]] = values

    def get_projection_equations(self, u_result, V_proj=None) -> tuple:
        if V_proj is None:
            try:
                V_proj = u_result.function_space
            except AttributeError:
                raise ValueError("V_proj is None and u_result is not a Function")

        v = TestFunction(V_proj)
        u_proj = TrialFunction(V_proj)

        dx_ = Measure("dx", domain=V_proj.mesh)

        a = inner(u_proj, v) * dx_
        L = inner(u_result, v) * dx_

        return a, L

    def get_linear_problem(self, a, L, bcs=[]):
        problem = LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options=self.linear_petsc_options,
        )
        return problem

    def get_nonlinear_problem(self, u, residual, bcs=[]):
        problem = NonlinearProblem(
            residual,
            u,
            bcs=bcs,
        )
        solver = NewtonSolver(MPI.COMM_WORLD, problem)

        solver.max_it = 100  # Increase max iterations
        solver.relaxation_parameter = 1.0  # Can reduce to 0.8 if needed
        solver.damping = 1.0  # Reduce the step size to stabilize convergence
        solver.ls = "bt"  # Use backtracking line search
        solver.convergence_criterion = "residual"
        solver.atol = 1e-6  # Absolute tolerance (adjust based on problem size)
        solver.rtol = 1e-6  # Relative tolerance
        ksp = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "gmres"
        opts[f"{option_prefix}pc_type"] = "gamg"
        # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
        ksp.setFromOptions()

        solver.solve = partial(solver.solve, u)

        return solver

    def check_export_results(self) -> bool:
        return self.step % 100 == 0

    def solve_time_step(self) -> None:
        self._solve_time_step()

        if self.check_export_results():
            for key, res in self._plot_variables().items():
                self.plot_results[key].append(res.x.array.copy())

    def advance_time(self) -> None:
        self.time += self.dt

    def run(self) -> None:
        self.setup()

        for self.step in progressbar(range(self.num_steps)):
            self.advance_time()

            self.solve_time_step()

        self.format_results()

    def format_results(self) -> None:
        for key, res in self.plot_results.items():
            var_func_space = self._plot_variables()[key].function_space
            nodal_func_space = functionspace(var_func_space.mesh, ("CG", 1, (2,)))

            res_variable = Function(var_func_space)

            problem = self.get_linear_problem(*self.get_projection_equations(res_variable, nodal_func_space))

            new_res = []
            for entry in res:
                res_variable.x.array[:] = entry
                new_res.append(problem.solve().x.array.copy())

            self.plot_results[key] = new_res

        self.formatted_plot_results = {key: format_vectors_from_flat(res, n_dim=self.dim) for key, res in self.plot_results.items()}

    def postprocess(self, scalars: str = None, vectors: str = None, scalar_process: str = None, name: str = "result", mesh=None) -> None:
        if mesh is None:
            mesh = self.mesh

        variables = {}
        variable_names = []
        if scalars is not None:
            if isinstance(scalars, str):
                scalar_variable = scalars
                variable_names.append(scalar_variable)
            else:
                scalar_variable = "scalar_variable"
                variables[scalar_variable] = scalars
        else:
            scalar_variable = None
            scalar_process = None

        if vectors is not None:
            if isinstance(vectors, str):
                vector_variable = vectors
                variable_names.append(vector_variable)
            else:
                vector_variable = "vector_variable"
                variables[vector_variable] = vectors
        else:
            vector_variable = None

        variable_names = set(variable_names)

        try:
            self.formatted_plot_results
        except AttributeError:
            self.format_results()

        for var in variable_names:
            if var in self.formatted_plot_results:
                variables[var] = self.formatted_plot_results[var]  # format_vectors_from_flat(self.plot_results[var], n_dim=self.dim)
            else:
                logger.warning(f"Variable {var} not found in plot results.")

        if scalar_process is None:
            scalar_value = variables.get(scalar_variable)
        elif scalar_process == "x":
            scalar_value = variables[scalar_variable][:, :, 0]
        elif scalar_process == "y":
            scalar_value = variables[scalar_variable][:, :, 1]
        elif scalar_process == "z":
            scalar_value = variables[scalar_variable][:, :, 2]
        elif scalar_process == "norm":
            scalar_value = np.linalg.norm(variables[scalar_variable], axis=-1)

        plot_mesh = vtk_mesh(mesh)
        create_mesh_animation(plot_mesh, scalar_value, variables.get(vector_variable), name=name)


class StructuralElasticSimulation(FenicsxSimulation):

    E = 200.0e3
    nu = 0.3
    rho = 7.85e-9

    def __init__(self) -> None:
        super().__init__()

    def _plot_variables(self) -> dict:
        return {
            "u": self.u_k,
        }

    def _solve_time_step(self) -> None:
        self.solve_u()

    def setup(self) -> None:
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lmbda = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.rho_dt = self.rho / self.dt**2

        self.G = self.E / (2 * (1 + self.nu))
        self.C = self.E / 5

        super().setup()

    def _define_functionspace(self) -> None:
        self.V = functionspace(self.mesh, (*self.element_type, (2,)))
        self.W = functionspace(self.mesh, (*self.element_type_sigma, (2, 2)))

    def _init_variables(self) -> None:
        self.u = TrialFunction(self.V)
        self.f = Constant(self.mesh, np.array([0.0, 0.0]))  # Force term

        self.u_k = Function(self.V, name="Displacement")
        self.u_prev = Function(self.V)
        self.u_next = Function(self.V)

        # Initialize displacement and acceleration to zero
        self.u_k.x.array[:] = 0.0
        self.u_prev.x.array[:] = 0.0

    @staticmethod
    def epsilon(u):
        return sym(grad(u))

    def sigma_elastic(self, u):
        epsilon = self.epsilon(u)
        return self.lmbda * tr(epsilon) * Identity(self.dim) + 2 * self.mu * epsilon

    def get_problem_equations(self, u, u_k, u_prev, f, sigma_func=None, **sigma_kwargs) -> tuple:
        if sigma_func is None:
            sigma_func = self.sigma_elastic

        V = u_k.function_space
        v = TestFunction(V)

        dx_ = Measure("dx", domain=V.mesh)
        stiffness_term = inner(sigma_func(u, **sigma_kwargs), self.epsilon(v)) * dx_
        mass_term = self.rho_dt * inner(u, v) * dx_
        a = mass_term + stiffness_term

        L_body = dot(f, v) * dx_
        L_mass = self.rho_dt * inner((2 * u_k - u_prev), v) * dx_
        L = self.apply_neumann_bcs(L_body + L_mass, V)

        return a, L, self.get_dirichlet_bcs(V)

    def get_traction_equations(self, u_t_next, u_t, u_t_prev, f_interface, ds_t, interface_marker_t, sigma_func=None, **sigma_kwargs) -> tuple:
        if sigma_func is None:
            sigma_func = self.sigma_elastic

        V_t = u_t_next.function_space
        v_t = TestFunction(V_t)

        dx_t = Measure("dx", domain=V_t.mesh)

        a_t = dot(f_interface, v_t) * ds_t(interface_marker_t) + dot(f_interface, v_t) * dx_t - dot(f_interface, v_t) * dx_t
        L_t = self.apply_neumann_bcs(inner(sigma_func(u_t_next, **sigma_kwargs), self.epsilon(v_t)) * dx_t + self.rho_dt * inner((u_t_next - 2 * u_t + u_t_prev), v_t) * dx_t, V_t)

        return a_t, L_t, self.get_dirichlet_bcs(V_t)

    def _define_differential_equations(self):

        self.elastic_problem = self.get_linear_problem(*self.get_problem_equations(self.u, self.u_k, self.u_prev, self.f))

    def _update_prev_values(self) -> None:
        self.u_prev.x.array[:] = self.u_k.x.array[:]
        self.u_k.x.array[:] = self.u_next.x.array[:]

    def solve_u(self) -> None:
        self.u_next = self.elastic_problem.solve()

    def solve_time_step(self) -> None:
        super().solve_time_step()

        self._update_prev_values()


class StructuralPlasticSimulation(StructuralElasticSimulation):
    sigma_yield = 250  # Yield stress (MPa)

    tol = 1e-6

    def _init_variables(self):
        super()._init_variables()

        self.zero_alpha = Function(self.W)
        self.zero_alpha.x.array[:] = 0.0

        self.Z = functionspace(self.mesh, self.element_type_sigma)
        self.yield_trial = TrialFunction(self.Z)
        self.yield_k = Function(self.Z)
        self.yield_k.x.array[:] = 0.0
        self.yield_diff = Function(self.Z)
        self.z = TestFunction(self.Z)

        self.alpha_k = Function(self.W)
        self.alpha_k.x.array[:] = 0.0
        self.alpha_trial = TrialFunction(self.W)
        self.w = TestFunction(self.W)
        self.alpha_projected = Function(self.W)

    def sigma_dev(self, sigma):
        return sigma - (1 / 3) * tr(sigma) * Identity(self.dim)

    def yield_function(self, sigma_dev, alpha):
        return sqrt(3 / 2 * inner(sigma_dev - alpha, sigma_dev - alpha)) - self.sigma_yield

    def yield_condition(self, sigma_dev, alpha):
        # return conditional(gt(self.yield_function(sigma_dev, alpha), self.tol), 1, 0)
        # return conditional(And(gt(self.yield_function(sigma_dev, alpha), self.tol), gt(self.yield_function(sigma_dev, alpha) - self.yield_k, self.tol)), 1, 0)
        return conditional(And(gt(self.yield_function(sigma_dev, alpha), self.tol), gt(self.yield_function(sigma_dev, alpha) - self.yield_function(self.sigma_dev(self.sigma_elastic(self.u_k)), alpha), self.tol)), 1, 0)
        # return conditional(gt((self.yield_function(sigma_dev, alpha) - self.yield_function(self.sigma_dev(self.sigma_elastic(self.u_k)), alpha) / 2), self.tol), 1, 0)

    def delta_epsilon(self, sigma_dev, alpha):
        norm = sqrt(inner(sigma_dev - alpha, sigma_dev - alpha))
        return conditional(gt(norm, self.tol), (3 / 2) * self.yield_function(sigma_dev, alpha) / (3 * self.G + self.C) * (sigma_dev - alpha) / norm, self.zero_alpha)

    def delta_alpha(self, sigma_dev, alpha):
        return (2 / 3) * self.C * self.delta_epsilon(sigma_dev, alpha)

    def delta_sigma(self, sigma_dev, alpha):
        return 2 * self.G * self.delta_epsilon(sigma_dev, alpha) + self.lmbda * tr(self.delta_epsilon(sigma_dev, alpha)) * Identity(self.dim)

    def alpha_next(self, u, alpha):
        sigma_dev = self.sigma_dev(self.sigma_elastic(u))
        return conditional(eq(self.yield_condition(sigma_dev, alpha), 1), alpha + self.delta_alpha(sigma_dev, alpha), alpha)

    def sigma_plastic(self, u, alpha):
        sigma_trial = self.sigma_elastic(u)
        sigma_dev_trial = self.sigma_dev(sigma_trial)
        return conditional(eq(self.yield_condition(sigma_dev_trial, alpha), 1), sigma_trial + self.delta_sigma(sigma_dev_trial, alpha), sigma_trial)

    def _define_differential_equations(self):
        a_sigma, L_sigma, bcs_sigma = self.get_problem_equations(self.u_next, self.u_k, self.u_prev, self.f, sigma_func=self.sigma_plastic, alpha=self.alpha_k)
        self.plastic_problem = self.get_nonlinear_problem(self.u_next, a_sigma - L_sigma, bcs_sigma)

        self.alpha_problem = self.get_linear_problem(*self.get_projection_equations(self.alpha_next(self.u_next, self.alpha_k), self.W))

        self.yield_problem = self.get_linear_problem(*self.get_projection_equations(self.yield_function(self.sigma_dev(self.sigma_elastic(self.u_next)), self.alpha_k), self.Z))

        self.yield_condition_problem = self.get_linear_problem(*self.get_projection_equations(self.yield_condition(self.sigma_dev(self.sigma_elastic(self.u_next)), self.alpha_k), self.Z))

    def solve_u(self) -> None:
        # yield_next = self.yield_problem.solve().x.array.copy()

        # self.yield_diff.x.array[:] = yield_next - self.yield_k.x.array[:]

        n, converged = self.plastic_problem.solve()

        # yield_condition = self.yield_condition_problem.solve().x.array.copy()

        # self.yield_k.x.array[:] = self.yield_problem.solve().x.array.copy()

        self.alpha_k.x.array[:] = self.alpha_problem.solve().x.array.copy()

        print(f"Solved step {self.step} in {n} iterations, ||u_|| = {np.linalg.norm(self.u_k.x.array):.2f}, ||alpha_|| = {np.linalg.norm(self.alpha_k.x.array):.2f}")

        # print(self.alpha_k.x.array[:40])
        # print(self.yield_k.x.array[:10])
        # print(yield_condition[:10])
        # print(self.yield_diff.x.array[:10])
        # print((self.yield_k.x.array > 0).mean())
        # print((self.yield_diff.x.array > 0).mean())

        # input("press enter")

    def check_export_results(self) -> bool:
        return self.step % 50 == 0


class StructuralPlasticSimulationTest(StructuralElasticSimulation):
    sigma_yield = 250  # Yield stress (MPa)

    tol = 1e-6

    def _init_variables(self):
        super()._init_variables()

        self.v = TestFunction(self.V)

        self.zero_alpha = Function(self.W)
        self.zero_alpha.x.array[:] = 0.0

        self.u_next_plastic = Function(self.V)
        self.u_next_plastic.x.array[:] = 0.0
        self.u_next_elastic = Function(self.V)

        self.u_mixed_trial = TrialFunction(self.V)
        self.u_mixed_k = Function(self.V)

        self.sigma_mixed = Function(self.W)

        self.u_sigma_plastic = TrialFunction(self.V)
        self.u_sigma_elastic = TrialFunction(self.V)

        self.Z = functionspace(self.mesh, self.element_type_sigma)
        self.yield_trial = TrialFunction(self.Z)
        self.yield_k = Function(self.Z)
        self.yield_k.x.array[:] = 0.0
        self.yield_diff = Function(self.Z)
        self.z = TestFunction(self.Z)

        self.alpha_k = Function(self.W)
        self.alpha_k.x.array[:] = 0.0
        self.alpha_trial = TrialFunction(self.W)
        self.w = TestFunction(self.W)
        self.alpha_projected = Function(self.W)

    def sigma_dev(self, sigma):
        return sigma - (1 / 3) * tr(sigma) * Identity(self.dim)

    def yield_function(self, sigma_dev, alpha):
        return sqrt(3 / 2 * inner(sigma_dev - alpha, sigma_dev - alpha)) - self.sigma_yield

    def yield_condition(self, sigma_dev, alpha):
        # return conditional(gt(self.yield_function(sigma_dev, alpha), self.tol), 1, 0)
        return conditional(And(gt(self.yield_function(sigma_dev, alpha), self.tol), gt(self.yield_function(sigma_dev, alpha) - self.yield_k, self.tol)), 1, 0)

    def delta_epsilon(self, sigma_dev, alpha):
        norm = sqrt(inner(sigma_dev - alpha, sigma_dev - alpha))
        return conditional(gt(norm, self.tol), (3 / 2) * self.yield_function(sigma_dev, alpha) / (3 * self.G + self.C) * (sigma_dev - alpha) / norm, self.zero_alpha)

    def delta_alpha(self, sigma_dev, alpha):
        return (2 / 3) * self.C * self.delta_epsilon(sigma_dev, alpha)

    def delta_sigma(self, sigma_dev, alpha):
        return 2 * self.G * self.delta_epsilon(sigma_dev, alpha) + self.lmbda * tr(self.delta_epsilon(sigma_dev, alpha)) * Identity(self.dim)

    def alpha_next(self, u, alpha):
        sigma_dev = self.sigma_dev(self.sigma_elastic(u))
        return alpha + self.delta_alpha(sigma_dev, alpha)

    def sigma_plastic(self, u, alpha):
        sigma_trial = self.sigma_elastic(u)
        sigma_dev_trial = self.sigma_dev(sigma_trial)
        return sigma_trial + self.delta_sigma(sigma_dev_trial, alpha)

    def _define_differential_equations(self):
        a_plastic, L_plastic, bcs_plastic = self.get_problem_equations(self.u_next_plastic, self.u_k, self.u_prev, self.f, sigma_func=self.sigma_plastic, alpha=self.alpha_k)
        self.plastic_problem = self.get_nonlinear_problem(self.u_next_plastic, a_plastic - L_plastic, bcs_plastic)

        a_elastic, L_elastic, bcs_elastic = self.get_problem_equations(self.u, self.u_k, self.u_prev, self.f, sigma_func=self.sigma_elastic)
        self.elastic_problem = self.get_linear_problem(a_elastic, L_elastic, bcs_elastic)

        self.sigma_plastic_problem = self.get_linear_problem(*self.get_projection_equations(self.sigma_plastic(self.u_next_plastic, self.alpha_k), self.W))
        self.sigma_elastic_problem = self.get_linear_problem(*self.get_projection_equations(self.sigma_elastic(self.u_next_elastic), self.W))

        a_u_problem = self.rho_dt * inner(self.u_mixed_trial, self.v) * dx
        b_u_problem = dot(self.f, self.v) * dx + self.rho_dt * inner((2 * self.u_k - self.u_prev), self.v) * dx - inner(self.sigma_mixed, self.epsilon(self.v)) * dx
        self.u_problem = self.get_linear_problem(a_u_problem, b_u_problem, self.get_dirichlet_bcs(self.V))

        self.alpha_problem = self.get_linear_problem(*self.get_projection_equations(self.alpha_next(self.u_k, self.alpha_k), self.W))

        self.yield_problem = self.get_linear_problem(*self.get_projection_equations(self.yield_condition(self.sigma_dev(self.sigma_elastic(self.u_next)), self.alpha_k), self.Z))

    def solve_u(self) -> None:
        # yield_next = self.yield_problem.solve().x.array.copy()

        # self.yield_diff.x.array[:] = yield_next - self.yield_k.x.array[:]

        print(self.alpha_k.x.array[:40])

        self.yield_k.x.array[:] = self.yield_problem.solve().x.array.copy()
        yielded_dofs = np.concatenate(np.dstack((self.yield_k.x.array[:], self.yield_k.x.array[:], self.yield_k.x.array[:], self.yield_k.x.array[:]))[0]).astype(np.int64)

        print(yielded_dofs)

        self.u_next_elastic.x.array[:] = self.elastic_problem.solve().x.array[:]  # solves u_next_elastic
        self.sigma_mixed.x.array[:] = self.sigma_elastic_problem.solve().x.array[:]
        if yielded_dofs.sum() > 0:
            print("updating yield")
            self.plastic_problem.solve()  # solves u_next_plastic
            self.alpha_k.x.array[yielded_dofs] = self.alpha_problem.solve().x.array[yielded_dofs]
            self.sigma_mixed.x.array[yielded_dofs] = self.sigma_plastic_problem.solve().x.array[yielded_dofs]

        self.u_next.x.array[:] = self.u_problem.solve().x.array[:]

        print(f"Solved step {self.step}, ||u_|| = {np.linalg.norm(self.u_k.x.array):.2f}, ||alpha_|| = {np.linalg.norm(self.alpha_k.x.array):.2f}")

        print(self.alpha_k.x.array[:40])
        print((self.yield_k.x.array > 0)[:10])
        # print(self.yield_diff.x.array[:10])
        print((self.yield_k.x.array > 0).mean())
        # print((self.yield_diff.x.array > 0).mean())

        # input("press enter")

    def check_export_results(self) -> bool:
        return self.step % 50 == 0
