import numpy as np
import pickle
import os

from tct.tct_tractions import TCTExtractTractions as extractor, TCTApplyTractions as applicator
from predictors.gradient_boosting import GradientBoosting
from predictors.lstm_network import LSTMNetwork, LSTMWindowNetwork


DATA_FOLDER = "results"


def generate_training_set(version: int = 1):
    frequency_range = range(500, 2001, int(1500 / 15))

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


def train_gradient_boosting(version: int = 1) -> None:
    with open(f"{DATA_FOLDER}/training_in_v{version:02}.npy", "rb") as f:
        training_in = np.load(f)

    with open(f"{DATA_FOLDER}/training_out_v{version:02}.npy", "rb") as f:
        training_out = np.load(f)

    reg = GradientBoosting(training_in, training_out)
    reg.fit()

    reg.save(f"{DATA_FOLDER}/model_v{version:02}.pkl")


def train_lstm(version: int = 1) -> None:
    with open(f"{DATA_FOLDER}/training_in_v{version:02}.npy", "rb") as f:
        training_in = np.load(f)

    with open(f"{DATA_FOLDER}/training_out_v{version:02}.npy", "rb") as f:
        training_out = np.load(f)

    reg = LSTMNetwork(training_in, training_out)
    reg.fit()

    reg.save(f"{DATA_FOLDER}/model_v{version:02}.pkl")
    print("Finished training model")


def apply_gradient_boosting(version: int = 1, frequency: int = 1000) -> None:
    predictor = GradientBoosting.load(f"{DATA_FOLDER}/model_v{version:02}.pkl")

    with open(f"{DATA_FOLDER}/model_forces.pkl", "rb") as f:
        model = pickle.load(f)

    predictor._model = model

    tct = applicator(predictor, frequency)
    tct.time_total = 2e-4
    tct.run()

    # tct.save(f"{DATA_FOLDER}/sim_results_v{version:02}.pkl")

    tct.postprocess("u", "u", "y", name=f"{DATA_FOLDER}/prediction_v{version:02}")


def apply_lstm(version: int = 1, frequency: int = 1000) -> None:
    predictor = LSTMNetwork.load(f"{DATA_FOLDER}/model_v{version:02}.pkl")

    predictor.initialize_memory_variables()
    tct = applicator(predictor, frequency)
    # tct.time_total = 2e-4
    tct.run()

    # tct.save(f"{DATA_FOLDER}/sim_results_v{version:02}.pkl")

    tct.postprocess("u", "u", "y", name=f"{DATA_FOLDER}/prediction_v{version:02}")


PREDICTOR_FUNCTIONS = {
    "gradient_boosting": (train_gradient_boosting, apply_gradient_boosting),
    "lstm": (train_lstm, apply_lstm)
}


def run(version: int, frequency: int = 1000, predictor_method: str = "lstm", simulate_only: bool = False, training_set_exists: bool = False) -> None:
    if not simulate_only:
        if not training_set_exists:
            generate_training_set(version)
        PREDICTOR_FUNCTIONS[predictor_method][0](version)

    PREDICTOR_FUNCTIONS[predictor_method][1](version, frequency)


if __name__ == "__main__":
    version = 9
    frequency = 1000
    predictor_method = "lstm"
    simulate_only = False
    training_set_exists = True

    try:
        os.mkdir(DATA_FOLDER)
    except FileExistsError:
        print("Folder already exists.")

    # train_predictor(version)

    run(version, frequency, predictor_method, simulate_only, training_set_exists)
