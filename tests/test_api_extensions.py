import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_drift_api_endpoint(client):
    res = client.get("/api/drift")
    assert res.status_code == 200
    data = res.json()
    assert "overall_status" in data
    assert "feature_drift" in data
    assert isinstance(data["feature_drift"], list)
    if len(data["feature_drift"]) > 0:
        first = data["feature_drift"][0]
        assert "feature" in first
        assert "psi" in first
        assert "drift_status" in first


def test_fairness_api_endpoint(client):
    res = client.get("/api/fairness")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    if len(data) > 0:
        item = data[0]
        assert "group" in item
        assert "fpr" in item
        assert "recall" in item


def test_ablation_api_endpoint(client):
    res = client.get("/api/ablation")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    if data["status"] == "available":
        assert "results" in data
        assert len(data["results"]) > 0
        first = data["results"][0]
        assert "feature_set" in first
        assert "model" in first
        assert "savings" in first
