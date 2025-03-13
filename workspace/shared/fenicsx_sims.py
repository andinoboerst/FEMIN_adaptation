import abc
import numpy as np
import logging
# import dill

from mpi4py import MPI
from dolfinx.fem import form, Constant, Expression, Function, functionspace, dirichletbc, locate_dofs_geometrical, locate_dofs_topological
from dolfinx.mesh import meshtags, locate_entities
from dolfinx.plot import vtk_mesh
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem, assemble_matrix, assemble_vector, create_matrix, _create_form, create_vector, assemble_matrix_mat
from dolfinx.nls.petsc import NewtonSolver
from ufl import TestFunction, TrialFunction, Identity, Measure, grad, inner, dot, tr, dx, sqrt, conditional, sym, derivative
from petsc4py import PETSc

from shared.plotting import format_vectors_from_flat, create_mesh_animation
from shared.progress_bar import progressbar

logger = logging.getLogger("fenicsx_sims")


class FenicsxSimulation(metaclass=abc.ABCMeta):
    # Time Stepping
    time_total = 3e-3
    dt = 5e-7

    element_type = "CG"

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

    def check_export_results(self) -> bool:
        return self.step % 100 == 0

    def solve_time_step(self) -> None:
        self._solve_time_step()

        if self.check_export_results():
            for key, res in self._plot_variables().items():
                self.plot_results[key].append(res)

    def advance_time(self) -> None:
        self.time += self.dt

    def run(self) -> None:
        self.setup()

        for self.step in progressbar(range(self.num_steps)):
            self.advance_time()

            self.solve_time_step()

        self.format_results()

    def format_results(self) -> None:
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

    # Newmark-beta parameters
    beta = 0.25
    gamma = 0.5

    linear_petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    def __init__(self) -> None:
        super().__init__()

    def _plot_variables(self) -> dict:
        return {
            "u": self.u_k.x.array.copy(),
        }

    def _solve_time_step(self) -> None:
        self.solve_u()

    def setup(self) -> None:
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lmbda = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.rho_dt = self.rho / self.dt**2

        super().setup()

    def _define_functionspace(self) -> None:
        self.V = functionspace(self.mesh, (self.element_type, 1, (2,)))
        self.W = functionspace(self.mesh, ("DG", 0, (2, 2)))

    def _init_variables(self) -> None:
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.f = Constant(self.mesh, np.array([0.0, 0.0]))  # Force term

        self.u_k = Function(self.V, name="Displacement")
        self.u_prev = Function(self.V)
        self.u_next = Function(self.V)

        self.sigma_projected = Function(self.W)
        self.w = TestFunction(self.W)
        self.tau = TrialFunction(self.W)

        # Initialize displacement and acceleration to zero
        self.u_k.x.array[:] = 0.0
        self.u_prev.x.array[:] = 0.0  # Important initialization

    def get_linear_equations(self, u, u_k, u_prev, f, v, V) -> tuple:
        dx_ = Measure("dx", domain=V.mesh)
        stiffness_term = inner(self.sigma(u), self.epsilon(v)) * dx_
        mass_term = self.rho_dt * inner(u, v) * dx_
        a = mass_term + stiffness_term

        L_body = dot(f, v) * dx_
        L_mass = self.rho_dt * inner((2 * u_k - u_prev), self.v) * dx_
        L = self.apply_neumann_bcs(L_body + L_mass, V)

        return a, L

    def solve_linear_problem(self, a, L, V):
        problem = LinearProblem(
            a,
            L,
            bcs=self.get_dirichlet_bcs(V),
            petsc_options=self.linear_petsc_options,
        )
        return problem.solve()

    def _define_differential_equations(self):

        self.a, self.L = self.get_linear_equations(self.u, self.u_k, self.u_prev, self.f, self.v, self.V)

    def _update_prev_values(self) -> None:
        self.u_prev.x.array[:] = self.u_k.x.array[:]
        self.u_k.x.array[:] = self.u_next.x.array[:]

    @staticmethod
    def epsilon(u):
        return sym(grad(u))

    def sigma(self, u):
        epsilon = self.epsilon(u)
        return self.lmbda * tr(epsilon) * Identity(self.dim) + 2 * self.mu * epsilon

    def solve_u(self) -> None:
        self.u_next = self.solve_linear_problem(self.a, self.L, self.V)

    def solve_time_step(self) -> None:
        super().solve_time_step()

        self._update_prev_values()


class StructuralPlasticSimulation(StructuralElasticSimulation):
    sigma_Y = 250e15  # Yield stress (MPa)
    C_k = 1  # Kinematic hardening modulus

    def sigma_dev(self, sigma):
        return sigma - (1 / 3) * tr(sigma) * Identity(2)

    def _init_variables(self):
        super()._init_variables()

        self.DG0 = functionspace(self.mesh, ("DG", 0))

        # self.sigma_old = Function(self.W)
        self.alpha_old = Function(self.W)
        self.eps_p_old = Function(self.W)

        self.sigma_correction_func = Function(self.W)

        self.f_trial_func = Function(self.DG0)
        self.delta_lambda_func = Function(self.DG0)
        self.sigma_trial_func = Function(self.W)
        self.s_trial_func = Function(self.W)

    def _define_differential_equations(self):
        # super()._define_differential_equations()

        # Define elastic trial stress
        self.sigma_trial = self.sigma(self.u_k)
        self.s_trial = self.sigma_dev(self.sigma_trial)

        # Compute yield function
        self.f_trial = sqrt(inner(self.s_trial - self.alpha_old, self.s_trial - self.alpha_old)) - self.sigma_Y

        # Return mapping (Plastic correction)
        self.delta_lambda = conditional(self.f_trial > 0, self.f_trial / (2 * self.mu + self.C_k), 0)

        norm_s = sqrt(inner(self.s_trial - self.alpha_old, self.s_trial - self.alpha_old))
        self.eps_p_correction = self.delta_lambda * (self.s_trial - self.alpha_old) / conditional(norm_s > 1e-8, norm_s, 1e-8)
        self.eps_p_new = self.eps_p_old + self.eps_p_correction

        self.alpha_correction = self.C_k * (self.eps_p_new - self.eps_p_old)
        self.alpha_new = self.alpha_old + self.alpha_correction

        # Compute updated stress
        self.sigma_correction = - 2 * self.mu * (self.eps_p_new - self.eps_p_old)
        self.sigma_new = self.sigma(self.u_k) + self.sigma_correction

        # Define variational problem
        # self.residual = self.apply_neumann_bcs(inner(self.f, self.v) * dx - inner(self.sigma_new, self.epsilon(self.v)) * dx)
        self.residual = inner(self.sigma(self.u_k), self.epsilon(self.v)) * dx

        # Define interpolations
        self.eps_p_expr = Expression(self.eps_p_new, self.W.element.interpolation_points())
        self.alpha_expr = Expression(self.alpha_new, self.W.element.interpolation_points())

        self.f_trial_expr = Expression(self.f_trial, self.DG0.element.interpolation_points())
        self.delta_lambda_expr = Expression(self.delta_lambda, self.DG0.element.interpolation_points())
        self.sigma_trial_expr = Expression(self.sigma_trial, self.W.element.interpolation_points())
        self.s_trial_expr = Expression(self.s_trial, self.W.element.interpolation_points())

        self.sigma_correction_expr = Expression(self.sigma_correction, self.sigma_correction_func.function_space.element.interpolation_points())

    def _update_prev_values(self) -> None:
        super()._update_prev_values()

        # Update values
        self.eps_p_old.interpolate(self.eps_p_expr)
        self.alpha_old.interpolate(self.alpha_expr)

    def bottom_displacement_function_new(self, t):
        return self.amplitude * np.sin(self.omega * t)

    def solve_u(self) -> None:

        self.sigma_correction_func.interpolate(self.sigma_correction_expr)
        self.f_trial_func.interpolate(self.f_trial_expr)
        self.delta_lambda_func.interpolate(self.delta_lambda_expr)
        self.sigma_trial_func.interpolate(self.sigma_trial_expr)
        self.s_trial_func.interpolate(self.s_trial_expr)
        # print("sigma_trial values: ", np.linalg.norm(self.sigma_trial_func.x.array))
        # print("s_trial values: ", np.linalg.norm(self.s_trial_func.x.array))
        # print("f_trial values: ", np.linalg.norm(self.f_trial_func.x.array))
        # print("delta_lambda values: ", np.linalg.norm(self.delta_lambda_func.x.array))
        # print("sigma correction values: ", np.linalg.norm(self.sigma_correction_func.x.array))

        # J = derivative(self.residual, self.u_k)

        # A = assemble_matrix(form(J), self.get_dirichlet_bcs())
        # A.assemble()

        # row_idx = 10  # Example row index
        # row_values = A.getRow(row_idx)
        # print(f"Row {row_idx} values: {row_values}")

        # ksp = PETSc.KSP().create(A.getComm())
        # ksp.setOperators(A)
        # ksp.setType("gmres")  # Use GMRES to analyze conditioning

        # # Compute and print condition number
        # A_norm = A.norm()
        # A_inv_norm = 1 / A_norm if A_norm > 1e-12 else np.inf  # Avoid division by zero
        # condition_number = A_norm * A_inv_norm
        # print(f"Condition number of Jacobian: {condition_number}")

        R_norm = np.linalg.norm(assemble_vector(form(self.residual)).array)
        print(f"Residual norm at iteration {self.step}: {R_norm}")

        # quit()

        problem = NonlinearProblem(self.residual, self.u_k, bcs=self.get_dirichlet_bcs())
        solver = NewtonSolver(MPI.COMM_WORLD, problem)

        solver.max_it = 10000  # Increase max iterations
        # solver.relaxation_parameter = 1.0  # Can reduce to 0.8 if needed
        # solver.damping = 1.0  # Reduce the step size to stabilize convergence
        # solver.ls = "bt"  # Use backtracking line search
        # solver.krylov_solver.setType("gmres")
        # solver.krylov_solver.getPC().setType("ilu")  # Algebraic multigrid
        # solver.convergence_criterion = "residual"
        # solver.krylov_solver = "gmres"  # More robust for nonlinear problems
        # solver.preconditioner = "ilu"   # Incomplete LU factorization for stability
        # solver.atol = 1e-6  # Absolute tolerance (adjust based on problem size)
        # solver.rtol = 1e-6  # Relative tolerance
        # ksp = solver.krylov_solver
        # opts = PETSc.Options()
        # option_prefix = ksp.getOptionsPrefix()
        # opts[f"{option_prefix}ksp_type"] = "gmres"
        # # opts[f"{option_prefix}pc_type"] = "gamg"
        # # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
        # ksp.setFromOptions()

        n, converged = solver.solve(self.u_k)

        print(f"Solver converged ({converged}) in {n} iterations")

        print(f"Solved step {self.step}, ||u_|| = {np.linalg.norm(self.u_k.x.array)}")

    def check_export_results(self) -> bool:
        return self.step % 1 == 0
