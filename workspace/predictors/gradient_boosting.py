import logging
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

from predictors.fenicsx_predictor import FenicsxPredictor


logger = logging.getLogger("GradientBoosting")


class GradientBoosting(FenicsxPredictor):

    def __init__(self, X: np.ndarray, Y: np) -> None:
        super().__init__(X.reshape(-1, X.shape[-1]), Y.reshape(-1, Y.shape[-1]))

        self.params = {
            "n_estimators": 500,
            "max_depth": 4,
            "min_samples_split": 5,
            "learning_rate": 0.01,
            "loss": "squared_error",
        }

    def fit(self) -> None:
        logger.info("Fitting data to predictor model...")

        self._model = MultiOutputRegressor(GradientBoostingRegressor(**self.params), n_jobs=5)
        self._model.fit(self.X_train, self.Y_train)

        mse = mean_squared_error(self.Y_test, self._model.predict(self.X_test))
        logger.info("The mean squared error (MSE) on test set: {:.4f}".format(mse))
