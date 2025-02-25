import numpy as np
import pickle
import os

from tct.tct_force import TCTForceExtract as extractor, TCTForceApply as applicator
from predictors.gradient_boosting import GradientBoosting


DATA_FOLDER = "results"


def generate_training_set(version: int = 1):
    frequency_range = range(500, 2001, int(1500 / 3))

    training_in = []
    training_out = []
    for frequency in frequency_range:
        print("Running Simulation for frequency: ", frequency)
        tct = extractor(frequency)
        tct.run()

        training_in.append(tct.data_in)
        training_out.append(tct.data_out)

    with open(f"{DATA_FOLDER}/training_in_v{version:02}.npy", "wb") as f:
        np.save(f, np.array(training_in))

    with open(f"{DATA_FOLDER}/training_out_v{version:02}.npy", "wb") as f:
        np.save(f, np.array(training_out))


def train_predictor(version: int = 1) -> None:
    with open(f"{DATA_FOLDER}/training_in_v{version:02}.npy", "rb") as f:
        training_in = np.load(f)

    with open(f"{DATA_FOLDER}/training_out_v{version:02}.npy", "rb") as f:
        training_out = np.load(f)
    
    reg = GradientBoosting(training_in, training_out)
    reg.fit()

    reg.save(f"{DATA_FOLDER}/model_v{version:02}.pkl")


def apply_predictor(version: int = 1, frequency: int = 1000) -> None:
    predictor = GradientBoosting.load(f"{DATA_FOLDER}/model_v{version:02}.pkl")

    with open(f"{DATA_FOLDER}/model_forces.pkl", "rb") as f:
        model = pickle.load(f)

    predictor._model = model

    tct = applicator(predictor, frequency)
    tct.time_total = 2e-4
    tct.run()

    # tct.save(f"{DATA_FOLDER}/sim_results_v{version:02}.pkl")

    tct.postprocess("u", "u", "y", name=f"{DATA_FOLDER}/prediction_v{version:02}")


def run(version: int, frequency: int = 1000, simulate_only: bool = False) -> None:
    if not simulate_only:
        generate_training_set(version)
        train_predictor(version)

    apply_predictor(version, frequency)


if __name__ == "__main__":
    version = 8
    frequency = 1000
    simulate_only = True

    try:
        os.mkdir(DATA_FOLDER)
    except FileExistsError:
        print("Folder already exists.")

    # train_predictor(version)
    
    run(version, frequency, simulate_only)
