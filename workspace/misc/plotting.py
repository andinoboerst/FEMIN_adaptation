import pyvista as pv
import pickle
import numpy as np
import matplotlib.pyplot as plt


def format_vectors_from_flat(u: np.ndarray, n_dim: int = 2) -> np.ndarray:
    u_x = []
    u_y = []
    u_z = []
    for u_i in u:
        u_x.append(u_i[::n_dim])
        u_y.append(u_i[1::n_dim])
        u_z.append(u_i[1::n_dim])

    if n_dim == 2:
        u_z = [[0] * len(u_x[0])] * len(u_x)

    return np.array([u_x, u_y, u_z]).transpose(1, 2, 0)



def create_mesh_animation(mesh, scalars=None, vectors=None, name: str="result", **kwargs) -> None:
    if scalars is None and vectors is None:
        raise ValueError("Either scalars or vectors must be provided")
    
    show_scalar_bar = True

    if scalars is None:
        show_scalar_bar = False
        scalars = np.zeros((len(vectors), len(vectors[0])))
    elif vectors is None:
        vectors = np.zeros((len(scalars), len(scalars[0]), 3))

    viridis = plt.get_cmap("viridis", 25)
    sargs = dict(
        title_font_size=25,
        label_font_size=20,
        fmt="%.2e",
        color="black",
        position_x=0.1,
        position_y=0.8,
        width=0.8,
        height=0.1,
    )

    grid_props = {
        "show_edges": True,
        "lighting": False,
        "cmap": viridis,
        "scalar_bar_args": sargs,
        "show_scalar_bar": show_scalar_bar,
        "clim": [scalars.min(), scalars.max()],
        "name": "grid",
    }

    pv.start_xvfb(0.5)  # Start virtual framebuffer for plotting

    grid_base = pv.UnstructuredGrid(*mesh)
    grid_base['scalars'] = scalars[0]
    grid_base.set_active_scalars('scalars')
    grid_base['vectors'] = vectors[0]
    grid = grid_base.warp_by_vector()


    plotter = pv.Plotter()
    plotter.open_gif(f"{name}.gif")

    plotter.add_mesh(
        grid,
        **grid_props,
    )

    plotter.view_xy()
    plotter.camera.zoom(1.3)

    for scalar, vector in zip(scalars, vectors):
        grid_base['vectors'] = vector
        grid = grid_base.warp_by_vector()
        grid['scalars'] = scalar
        plotter.add_mesh(
            grid,
            **grid_props
        )
        plotter.write_frame()

    plotter.close()