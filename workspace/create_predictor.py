import numpy as np
import pickle

from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from tct.tct_elastic import tct_elastic_generate_u_interface, tct_elastic_apply_u_interface
from misc.plotting import format_vectors_from_flat, create_mesh_animation


DATA_FOLDER = "results"


def generate_training_set(version: int = 1):
    frequency_range = range(500, 2001, int(1500/5))

    training_in = []
    training_out = []
    for frequency in frequency_range:
        print("Running Simulation for frequency: ", frequency)
        u, t = tct_elastic_generate_u_interface(frequency)
    
        training_in.append(t)
        training_out.append(u)

    with open(f"{DATA_FOLDER}/training_in_v{version:02}.npy", "wb") as f:
        np.save(f, np.array(training_in))

    with open(f"{DATA_FOLDER}/training_out_v{version:02}.npy", "wb") as f:
        np.save(f, np.array(training_out))


def train_predictor(version: int = 1) -> None:
    with open(f"{DATA_FOLDER}/training_in_v{version:02}.npy", "rb") as f:
        training_in = np.load(f)

    with open(f"{DATA_FOLDER}/training_out_v{version:02}.npy", "rb") as f:
        training_out = np.load(f)

    training_in = training_in.reshape(-1, 42)
    training_out = training_out.reshape(-1, 42)

    X_train, X_test, y_train, y_test = train_test_split(
        training_in, training_out, test_size=0.2, random_state=13
    )

    params = {
        "n_estimators": 500,
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "squared_error",
    }

    reg = MultiOutputRegressor(ensemble.GradientBoostingRegressor(**params), n_jobs=-1)
    reg.fit(X_train, y_train)

    mse = mean_squared_error(y_test, reg.predict(X_test))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

    with open(f"{DATA_FOLDER}/model_v{version:02}.pkl", "wb") as f:
        pickle.dump(reg, f)


def apply_predictor(version: int = 1, frequency: int = 1000) -> None:
    with open(f"{DATA_FOLDER}/model_v{version:02}.pkl", "rb") as f:
        predictor = pickle.load(f)
    
    mesh, u, v = tct_elastic_apply_u_interface(predictor, frequency)

    u_tensor = format_vectors_from_flat(u)
    v_tensor = format_vectors_from_flat(v)

    with open(f"{DATA_FOLDER}/sim_results_v{version:02}.pkl", "wb") as f:
        pickle.dump((mesh, u_tensor, v_tensor), f)

    create_mesh_animation(mesh, u_tensor[:, :, 1], u_tensor, name=f"{DATA_FOLDER}/prediction_v{version:02}")


def run(version: int, frequency: int = 1000, simulate_only: bool = False) -> None:
    if not simulate_only:
        generate_training_set(version)
        train_predictor(version)

    apply_predictor(version, frequency)


if __name__ == "__main__":
    version = 4
    frequency = 1000
    simulate_only = False
    
    run(version, frequency, simulate_only)
