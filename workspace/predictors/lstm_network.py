import numpy as np

from predictors.fenicsx_predictor import FenicsxPredictor


class LSTMNetwork(FenicsxPredictor):
    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        super().__init__(X, Y)