import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston


model_path = Path(__file__).parent / 'model.joblib'
n_features = load_boston(return_X_y=True)[0].shape[1]


def train_model(X, y):
    model = RandomForestRegressor()
    model.fit(X, y)
    return model


def save_model(model):
    joblib.dump(model, model_path)


def load_model():
    model = joblib.load(model_path)
    return model


if __name__ == "__main__":
    X, y = load_boston(return_X_y=True)
    model = train_model(X, y)
    save_model(model)
