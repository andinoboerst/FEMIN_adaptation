import abc
import numpy as np
import logging
from functools import partial

from mpi4py import MPI
from dolfinx.fem import Constant, Function, functionspace, dirichletbc, locate_dofs_topological
from dolfinx.mesh import meshtags, locate_entities
from dolfinx.plot import vtk_mesh
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from ufl import TestFunction, TrialFunction, Identity, Measure, grad, inner, dot, tr, sqrt, conditional, sym, gt, eq, And, replace, lhs, rhs
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

    def get_nodes(self, marker, V=None, points: bool = True) -> np.array:
        if V is None:
            V = self.V

        mesh = V.mesh
        mesh_dim = mesh.topology.dim

        if points:
            fdim = 0
        else:
            fdim = mesh_dim

        coords = np.around(V.tabulate_dof_coordinates(), decimals=3)

        # Find the facets on the top boundary
        entities = locate_entities(mesh, fdim, marker)

        mesh.topology.create_connectivity(fdim, 2)
        nodes = locate_dofs_topological(V, fdim, entities)

        coords_dtype = coords.dtype
        dt = [('x', coords_dtype), ('y', coords_dtype), ('z', coords_dtype)]
        ind = np.argsort(coords[nodes].ravel().view(dt), order=['x', 'y', 'z'])
        return nodes[ind]

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
                self._applied_dirichlet_bcs[0].append(V.mesh)
                self._applied_dirichlet_bcs[1].append([])

            i = self._applied_dirichlet_bcs[0].index(V.mesh)
            self._applied_dirichlet_bcs[1][i].append(dirichletbc(func, nodes))

    def get_dirichlet_bcs(self, mesh=None):
        if mesh is None:
            mesh = self.mesh

        try:
            index = self._applied_dirichlet_bcs[0].index(mesh)
            return self._applied_dirichlet_bcs[1][index]
        except ValueError:
            return []

    def _setup_neumann_bcs(self) -> None:
        self._neumann_bcs, facet_info = {}, {}
        for boundary, marker, V in self._neumann_bcs_list:
            mesh = V.mesh
            fdim = mesh.topology.dim - 1
            dofs = self.get_dofs(self.get_nodes(boundary, V))
            self._neumann_bcs[marker] = (Function(V), dofs)
            facets = locate_entities(mesh, fdim, boundary)
            if mesh not in facet_info:
                facet_info[mesh] = (
                    [],
                    [],
                    [],
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

            self._applied_neumann_bcs[0].append(mesh)
            self._applied_neumann_bcs[1].append(L)

    def apply_neumann_bcs(self, L, mesh=None):
        if mesh is None:
            mesh = self.mesh

        try:
            index = self._applied_neumann_bcs[0].index(mesh)
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

    def get_projection_problem(self, u_projected, u_result) -> tuple:
        V_proj = u_projected.function_space

        v = TestFunction(V_proj)
        du = TrialFunction(V_proj)

        dx_ = Measure("dx", domain=V_proj.mesh)

        a = inner(du, v) * dx_
        L = inner(u_result, v) * dx_

        return self.get_linear_problem(u_projected, a - L)

    def get_linear_problem(self, u, residual, bcs=[]):
        V = u.function_space
        du = TrialFunction(V)
        Residual_du = replace(residual, {u: du})
        a_form = lhs(Residual_du)
        L_form = rhs(Residual_du)

        problem = LinearProblem(
            a_form,
            L_form,
            bcs=bcs,
            petsc_options=self.linear_petsc_options,
        )

        class LinearSolver:
            def __init__(self, u, problem) -> None:
                self.u = u
                self.problem = problem

            def solve(self):
                self.u.x.array[:] = self.problem.solve().x.array[:]
        return LinearSolver(u, problem)

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

    def update_prev_values(self) -> None:
        pass

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
            proj_variable = Function(nodal_func_space)

            problem = self.get_projection_problem(proj_variable, res_variable)

            new_res = []
            for entry in res:
                res_variable.x.array[:] = entry
                problem.solve()
                new_res.append(proj_variable.x.array.copy())

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


class StructuralSimulation(FenicsxSimulation):

    E = 200.0e3
    nu = 0.3
    rho = 7.85e-9
    sigma_yield = 250  # Yield stress (MPa)

    body_force = np.array([0.0, 0.0])

    tol = 1e-6

    beta = 0.25
    gamma = 0.5

    constitutive_model_options = ["elastic", "plastic"]
    constitutive_model = "elastic"

    def __init__(self) -> None:
        super().__init__()

        if self.constitutive_model not in self.constitutive_model_options:
            raise ValueError(f"Unknown constitutive model: {self.constitutive_model}. Needs to be one of {self.constitutive_model_options}.")

    def _plot_variables(self) -> dict:
        return {
            "u": self.u_k,
        }

    def setup(self) -> None:
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lmbda = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        self.G = self.E / (2 * (1 + self.nu))
        self.C = self.E / 5

        super().setup()

    def _define_functionspace(self) -> None:
        self.V = functionspace(self.mesh, (*self.element_type, (2,)))
        self.W = functionspace(self.mesh, (*self.element_type_sigma, (2, 2)))

    def _init_variables(self) -> None:
        self.f = Constant(self.mesh, self.body_force)  # Force term

        self.u_k = Function(self.V, name="Displacement")
        self.v_k = Function(self.V, name="Velocity")
        self.a_k = Function(self.V, name="Acceleration")

        self.u_next = Function(self.V)
        self.v_next = Function(self.V)
        self.a_next = Function(self.V)

        # Initialize variables to zero
        self.u_k.x.array[:] = 0.0
        self.v_k.x.array[:] = 0.0
        self.a_k.x.array[:] = 0.0

        # Plastic variables
        self.alpha_k = Function(self.W)
        self.alpha_next = Function(self.W)
        self.alpha_k.x.array[:] = 0.0

    def setup_traction_problem(
        self,
        mesh_t,
        mesh,
        function_overlapping_points,
        function_overlapping_elements,
        interface_function,
    ) -> None:
        V = functionspace(mesh, (*self.element_type, (2,)))
        W = functionspace(mesh, (*self.element_type_sigma, (2, 2)))

        V_t = functionspace(mesh_t, (*self.element_type, (2,)))
        W_t = functionspace(mesh_t, (*self.element_type_sigma, (2, 2)))

        self.f_res = Function(V_t)
        self.u_next_t = Function(V_t)
        self.u_k_t = Function(V_t)
        self.v_k_t = Function(V_t)
        self.a_k_t = Function(V_t)
        self.f_t = Constant(mesh_t, self.body_force)  # Force term
        self.alpha_k_t = Function(W_t)

        # Full simulation
        self.overlapping_nodes = self.get_nodes(function_overlapping_points, V=V)
        self.overlapping_elements_sigma = self.get_dofs(self.get_nodes(function_overlapping_elements, V=W, points=False), value_dim=2)
        self.overlapping_dofs = self.get_dofs(self.overlapping_nodes)

        # Traction extraction
        self.overlapping_nodes_t = self.get_nodes(function_overlapping_points, V=V_t)
        self.overlapping_elements_sigma_t = self.get_dofs(self.get_nodes(function_overlapping_elements, V=W_t, points=False), value_dim=2)
        self.overlapping_dofs_t = self.get_dofs(self.overlapping_nodes_t)

        interface_nodes_t = self.get_nodes(interface_function, V=V_t)
        self.interface_dofs_t = self.get_dofs(interface_nodes_t)

        interface_marker_t = 88

        fdim = mesh_t.topology.dim - 1
        facets = locate_entities(mesh_t, fdim, interface_function).astype(np.int32)
        facets_marker = np.full_like(facets, interface_marker_t).astype(np.int32)
        facet_tag = meshtags(mesh_t, fdim, facets, facets_marker)

        ds_t = Measure("ds", domain=mesh_t, subdomain_data=facet_tag)(interface_marker_t)

        self.add_dirichlet_bc(lambda x: ~interface_function(x), 1234, V_t)

        return self.f_res, self.u_next_t, self.u_k_t, self.v_k_t, self.a_k_t, self.f_t, ds_t, self.constitutive_model, self.alpha_k_t

    @staticmethod
    def epsilon(u):
        return sym(grad(u))

    def sigma_elastic(self, u):
        epsilon = self.epsilon(u)
        return self.lmbda * tr(epsilon) * Identity(self.dim) + 2 * self.mu * epsilon

    def sigma_dev(self, sigma):
        return sigma - (1 / 3) * tr(sigma) * Identity(self.dim)

    def yield_function(self, sigma_dev, alpha):
        return sqrt(3 / 2 * inner(sigma_dev - alpha, sigma_dev - alpha)) - self.sigma_yield

    def yield_condition(self, sigma_dev, alpha):
        return conditional(gt(self.yield_function(sigma_dev, alpha), self.tol), 1, 0)
        # return conditional(And(gt(self.yield_function(sigma_dev, alpha), self.tol), gt(self.yield_function(sigma_dev, alpha) - self.yield_function(self.sigma_dev(self.sigma_elastic(self.u_k)), alpha), self.tol)), 1, 0)

    def delta_epsilon(self, sigma_dev, alpha):
        zero_alpha = Function(alpha.function_space)
        zero_alpha.x.array[:] = 0.0
        norm = sqrt(inner(sigma_dev - alpha, sigma_dev - alpha))
        return conditional(gt(norm, self.tol), (3 / 2) * self.yield_function(sigma_dev, alpha) / (3 * self.G + self.C) * (sigma_dev - alpha) / norm, zero_alpha)

    def delta_alpha(self, sigma_dev, alpha):
        return (2 / 3) * self.C * self.delta_epsilon(sigma_dev, alpha)

    def delta_sigma(self, sigma_dev, alpha):
        return 2 * self.G * self.delta_epsilon(sigma_dev, alpha) + self.lmbda * tr(self.delta_epsilon(sigma_dev, alpha)) * Identity(self.dim)

    def alpha_n_plus_one(self, u, alpha):
        sigma_dev = self.sigma_dev(self.sigma_elastic(u))
        return conditional(eq(self.yield_condition(sigma_dev, alpha), 1), alpha + self.delta_alpha(sigma_dev, alpha), alpha)

    def sigma_plastic(self, u, alpha):
        sigma_trial = self.sigma_elastic(u)
        sigma_dev_trial = self.sigma_dev(sigma_trial)
        return conditional(eq(self.yield_condition(sigma_dev_trial, alpha), 1), sigma_trial + self.delta_sigma(sigma_dev_trial, alpha), sigma_trial)

    def velocity(self, a_next, v_k, a_k):
        return v_k + (1 - self.gamma) * self.dt * a_k + self.gamma * self.dt * a_next

    def acceleration(self, u_next, u_k, v_k, a_k):
        return (1 / self.beta) * ((self.beta - 0.5) * a_k + (1 / self.dt**2) * (u_next - u_k - self.dt * v_k))

    def get_constitutive_functions(self, constitutive_model, alpha_k) -> tuple:
        if constitutive_model == "elastic":
            sigma_func = self.sigma_elastic
            get_problem_func = self.get_linear_problem
            sigma_kwargs = {}
        elif constitutive_model == "plastic":
            if alpha_k is None:
                raise ValueError("alpha_k must be provided for plastic consitutive model.")
            sigma_func = self.sigma_plastic
            get_problem_func = self.get_nonlinear_problem
            sigma_kwargs = {"alpha": alpha_k}

        return sigma_func, get_problem_func, sigma_kwargs

    def get_problem_equations(self, u_next, u_k, v_k, a_k, f, sigma_func, **sigma_kwargs) -> tuple:
        V = u_next.function_space
        v = TestFunction(V)
        dx_ = Measure("dx", domain=V.mesh)

        stiffness_term = inner(sigma_func(u_next, **sigma_kwargs), self.epsilon(v)) * dx_
        mass_term = self.rho * inner(self.acceleration(u_next, u_k, v_k, a_k), v) * dx_
        a = mass_term + stiffness_term

        L_body = dot(f, v) * dx_
        L = self.apply_neumann_bcs(L_body, V.mesh)

        return u_next, a - L, self.get_dirichlet_bcs(V.mesh)

    def get_traction_problem(self, f_interface, u_t_next, u_t_k, v_t_k, a_t_k, f_t, ds_t, constitutive_model: str = None, alpha_k=None) -> tuple:
        sigma_func, _, sigma_kwargs = self.get_constitutive_functions(constitutive_model, alpha_k)

        V_t = u_t_next.function_space
        v_t = TestFunction(V_t)

        dx_t = Measure("dx", domain=V_t.mesh)

        u_t_next, residual, bcs = self.get_problem_equations(u_t_next, u_t_k, v_t_k, a_t_k, f_t, sigma_func, **sigma_kwargs)

        residual -= dot(f_interface, v_t) * ds_t + dot(f_interface, v_t) * dx_t - dot(f_interface, v_t) * dx_t

        return self.get_linear_problem(f_interface, residual, bcs)

    def get_main_problems(self, u_next, v_next, a_next, u_k, v_k, a_k, f, constitutive_model: str = None, alpha_k=None, alpha_next=None):
        if alpha_k is not None and alpha_next is None:
            raise ValueError("alpha_next must be provided if alpha_k is provided.")

        sigma_func, get_problem_func, sigma_kwargs = self.get_constitutive_functions(constitutive_model, alpha_k)

        u_problem = get_problem_func(*self.get_problem_equations(u_next, u_k, v_k, a_k, f, sigma_func, **sigma_kwargs))

        acceleration_problem = self.get_projection_problem(a_next, self.acceleration(u_next, u_k=u_k, v_k=v_k, a_k=a_k))
        velocity_problem = self.get_projection_problem(v_next, self.velocity(a_next=a_next, v_k=v_k, a_k=a_k))

        if constitutive_model == "elastic":
            return u_problem, acceleration_problem, velocity_problem
        else:
            alpha_problem = self.get_projection_problem(alpha_next, self.alpha_n_plus_one(u_next, alpha_k))
            return u_problem, acceleration_problem, velocity_problem, alpha_problem

    def calculate_interface_tractions(self) -> None:
        self.alpha_k_t.x.array[self.overlapping_elements_sigma_t] = self.alpha_k.x.array[self.overlapping_elements_sigma].copy()
        self.u_next_t.x.array[self.overlapping_dofs_t] = self.u_next.x.array[self.overlapping_dofs].copy()
        self.u_k_t.x.array[self.overlapping_dofs_t] = self.u_k.x.array[self.overlapping_dofs].copy()
        self.v_k_t.x.array[self.overlapping_dofs_t] = self.v_k.x.array[self.overlapping_dofs].copy()
        self.a_k_t.x.array[self.overlapping_dofs_t] = self.a_k.x.array[self.overlapping_dofs].copy()

        self.traction_problem.solve()

        return self.f_res.x.array[self.interface_dofs_t].copy()

    def _define_differential_equations(self):
        self.main_problems = self.get_main_problems(self.u_next, self.v_next, self.a_next, self.u_k, self.v_k, self.a_k, self.f, self.constitutive_model, self.alpha_k, self.alpha_next)

    def solve_u(self) -> None:

        self.update_prev_values()

        for problem in self.main_problems:
            problem.solve()

        # print(f"Solved step {self.step}, ||u_|| = {np.linalg.norm(self.u_next.x.array):.2f}, ||alpha_|| = {np.linalg.norm(self.alpha_next.x.array):.2f}")

    def update_prev_values(self) -> None:
        self.u_k.x.array[:] = self.u_next.x.array[:]
        self.v_k.x.array[:] = self.v_next.x.array[:]
        self.a_k.x.array[:] = self.a_next.x.array[:]

        self.alpha_k.x.array[:] = self.alpha_next.x.array[:]

    def _solve_time_step(self) -> None:
        self.solve_u()
