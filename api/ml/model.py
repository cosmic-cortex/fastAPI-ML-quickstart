import joblib
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston


model_path = Path(__file__).parent / "model.joblib"
n_features = load_boston(return_X_y=True)[0].shape[1]


class Model:
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model = RandomForestRegressor()
        self.model.fit(X, y)
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self):
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
        else:
            raise TypeError("The model is not trained yet, use .train() before saving")

    def load(self):
        self.model = joblib.load(self.model_path)


model = Model(model_path)


if __name__ == "__main__":
    X, y = load_boston(return_X_y=True)
    model.train(X, y)
    model.save()
