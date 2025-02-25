import abc
import numpy as np
import pickle

from sklearn.model_selection import train_test_split


class FenicsxPredictor(metaclass=abc.ABCMeta):

    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X_base = X
        self.Y_base = Y

        self._train_test_split()

    @abc.abstractmethod
    def fit(self) -> None:
        raise NotImplementedError("Need to implement fit().")

    def _train_test_split(self) -> None:
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X_base, self.Y_base, test_size=0.4, random_state=13
        )

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict([x])[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
    
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "FenicsxPredictor":
        with open(path, "rb") as f:
            return pickle.load(f)
