from fastapi.testclient import TestClient
from app import app
client=TestClient(app)
def test_health():
    r=client.get("/health")
    assert r.status_code==200 and "model_available" in r.json()
def test_prediction_requires_model_or_returns_score():
    r=client.post("/predict",json={"amount":100,"hour":12,"distance_km":2,"failed_attempts":0})
    assert r.status_code in (200,503)
