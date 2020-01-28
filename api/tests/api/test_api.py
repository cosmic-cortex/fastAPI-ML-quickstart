import pytest
import random
import tempfile
from pathlib import Path
from itertools import product

from starlette.testclient import TestClient
from starlette.status import HTTP_200_OK, HTTP_422_UNPROCESSABLE_ENTITY

from api.ml.model import n_features


@pytest.mark.parametrize("n_instances", range(1, 10))
def test_predict(n_instances: int, test_client: TestClient):
    fake_data = [[random.random() for _ in range(n_features)] for _ in range(n_instances)]
    response = test_client.post("/predict", json={"data": fake_data})
    assert response.status_code == HTTP_200_OK
    assert len(response.json()["data"]) == n_instances


@pytest.mark.parametrize(
    "n_instances, test_data_n_features",
    product(range(1, 10), [n for n in range(1, 20) if n != n_features]),
)
def test_predict_with_wrong_input(
    n_instances: int, test_data_n_features: int, test_client: TestClient
):
    fake_data = [[random.random() for _ in range(test_data_n_features)] for _ in range(n_instances)]
    response = test_client.post("/predict", json={"data": fake_data})
    assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY


def test_predict_csv(test_client: TestClient):
    data_path = Path(__file__).parent / "data_correct.csv"
    with open(data_path, "r") as csv_file:
        response = test_client.post("/predict_csv", files={"csv_file": csv_file})
        assert response.status_code == HTTP_200_OK


def test_predict_csv_with_wrong_input(test_client: TestClient):
    data_path = Path(__file__).parent / "data_incorrect.csv"
    with open(data_path, "r") as csv_file:
        response = test_client.post("/predict_csv", files={"csv_file": csv_file})
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY


def test_predict_csv_with_noncsv_file(test_client: TestClient):
    with open(__file__, "r") as file:
        response = test_client.post("/predict_csv", files={"csv_file": file})
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY
