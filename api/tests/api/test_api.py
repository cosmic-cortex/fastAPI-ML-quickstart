import pytest
import random
from starlette.testclient import TestClient
from fastapi.encoders import jsonable_encoder

from api.main import PredictRequest
from api.ml.model import n_features


@pytest.mark.parametrize("n_instances", range(1, 10))
def test_predict(n_instances: int, test_client: TestClient):
    fake_data = [[random.random() for _ in range(n_features)]
                 for _ in range(n_instances)]
    json_data = jsonable_encoder(PredictRequest(data=fake_data))
    response = test_client.post('/predict', json=json_data)
    assert response.status_code == 200
    assert len(response.json()['data']) == n_instances
