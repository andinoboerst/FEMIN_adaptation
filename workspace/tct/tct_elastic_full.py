import numpy as np
import pickle

# import dolfinx
from mpi4py import MPI
from dolfinx.fem import Constant, Function, functionspace, dirichletbc, locate_dofs_geometrical
from dolfinx.mesh import create_rectangle, CellType
from dolfinx.plot import vtk_mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import TestFunction, TrialFunction, Identity, grad, inner, tr, dx
from dolfinx.io import XDMFFile, VTKFile

from misc.progress_bar import progressbar


def tct_elastic_full() -> None:
    # 1. Domain and Mesh
    width = 100.0
    height = 50.0
    element_size_x = 5.0
    element_size_y = 5.0
    nx = int(width / element_size_x)
    ny = int(height / element_size_y)

    mesh = create_rectangle(MPI.COMM_WORLD, cell_type=CellType.quadrilateral,
                            points=((0.0, 0.0), (width, height)), n=(nx, ny))

    # 2. Function Space
    V = functionspace(mesh, ("CG", 1, (2,)))
    W = functionspace(mesh, ("DG", 0, (2, 2)))

    # 3. Material Properties (Linear Elasticity)
    E = 200.0e3  # Young's modulus
    nu = 0.3   # Poisson's ratio
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # 4. Boundary Conditions
    def top_boundary(x):
        return np.isclose(x[1], height)

    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)

    def interface_boundary(x):
        return np.isclose(x[1], 25.0)

    top_boundary_nodes = locate_dofs_geometrical(V, top_boundary)
    bottom_boundary_nodes = locate_dofs_geometrical(V, bottom_boundary)
    interface_boundary_nodes = locate_dofs_geometrical(V, interface_boundary)

    # Top BC: Fixed (Dirichlet BC)
    bc_top = dirichletbc(np.array([0.0, 0.0], dtype=np.float64), top_boundary_nodes, V) # Use top_boundary function
    bcs = [bc_top]

    # Bottom BC: Sinusoidal displacement (Time-dependent)
    amplitude = 5.0
    frequency = 1000/3 # Hz, approx one cycle in 0.003s
    omega = 2 * np.pi * frequency

    # Define displacement function for bottom boundary - will be updated in time loop
    def bottom_displacement_function(t):
        value = amplitude * np.sin(omega * t)
        return Constant(mesh, np.array([0, value]))


    def epsilon(u):
        return 0.5 * (grad(u) + grad(u).T)

    def sigma(u):
        return lmbda * tr(epsilon(u)) * Identity(mesh.geometry.dim) + 2 * mu * epsilon(u)

    u = TrialFunction(V)
    v = TestFunction(V)

    f = Function(V)
    f.x.array[:] = 0.0

    a = inner(sigma(u), epsilon(v)) * dx  # Use dx_ufl
    L = inner(f, v) * dx        # Use dx_ufl

    # Time Stepping
    time_total = 0.003 # ms
    dt = 5e-7 # ms
    num_steps = int(time_total / dt)
    time = 0.0

    tdim = mesh.topology.dim

    # Initialize solution Function
    u_k = Function(V, name="Displacement")
    u_prev = Function(V) # Function to store displacement from previous time step
    velocity_k = Function(V, name="Velocity") # Function to store velocity
    # sigma_h = Function(W)
    # n = Constant(mesh, np.array([0.0, -1.0]))

    # Initialize u_prev to zero
    u_prev.x.array[:] = 0.0

    u_full = []
    v_full = []
    for step in progressbar(range(num_steps)):
        time += dt

        # Update top boundary condition
        bc_top = dirichletbc(np.array([0.0, 0.0], dtype=np.float64), top_boundary_nodes, V)

        # Update bottom boundary condition
        bottom_displacement_expr = bottom_displacement_function(time)

        bc_bottom = dirichletbc(bottom_displacement_expr, # Use u_bottom_func again
                                bottom_boundary_nodes, V)

        current_bcs = [bc_top, bc_bottom]

        # Solve linear elasticity problem
        problem = LinearProblem(a, L, bcs=current_bcs, u=u_k)
        u_k = problem.solve()

        velocity_k.x.array[:] = (u_k.x.array[:] - u_prev.x.array[:]) / dt

        # --- Calculate Stress and Extract at x=25 (using INITIAL node locations) ---
        # sigma_expr = sigma(u_k)
        # sigma_project = form(inner(sigma_expr, TestFunction(W)) * dx)
        # problem_stress = LinearProblem(form(inner(TestFunction(W), TrialFunction(W)) * dx), sigma_project, u=sigma_h)
        # problem_stress.solve()

        # T_interface = []
        # print(f"\n--- Stress at Nodes initially at x=25 at Time {time:.5f} ---")
        # for node_index in interface_boundary_nodes:
        #     current_coords = mesh.geometry.x[node_index]
        #     cells = locate_entities(mesh, tdim, interface_boundary)
        #     stress_value = sigma_h.eval(current_coords, cells)
        #     sigma_xx = stress_value[0]
        #     sigma_yy = stress_value[3]
        #     sigma_xy = stress_value[1]

        #     t_x = sigma_xx * n.value[0] + sigma_xy * n.value[1]
        #     t_y = sigma_xy * n.value[0] + sigma_yy * n.value[1]

        #     T_interface.append([t_x, t_y])
        
        # print(T_interface)

        # Save DEFORMED MESH and fields
        if step % 100 == 0:
            u_full.append(u_k.x.array.copy())
            v_full.append(velocity_k.x.array.copy())

        # --- Update u_prev for next time step ---
        u_prev.x.array[:] = u_k.x.array[:] # Copy current displacement to u_prev for next velocity calculation

    print("Simulation complete")

    return vtk_mesh(mesh), u_full, v_full

    # with open("u_full.npy", "wb") as f:
    #     np.save(f, u_full)

    # with open("v_full.npy", "wb") as f:
    #     np.save(f, v_full)


if __name__ == "__main__":
    tct_elastic_full()