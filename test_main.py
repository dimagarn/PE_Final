from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {'model': 'Text Classifier'}


def test_predict_positive():
    response = client.post("/predict/",
                           json={"text": "I like cats!"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data[0]["label"] == 'love'


def test_predict_negative():
    response = client.post("/predict/",
                           json={"text": "I hate machine learning!"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data[0]["label"] == 'anger'

def test_predict_desire():
    response = client.post("/predict/",
                           json={"text": "I want pizza"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data[0]["label"] == 'desire'