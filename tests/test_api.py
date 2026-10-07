import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import api


@pytest.fixture
def client(monkeypatch):
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=200)),
    ])
    model.fit(
        [
            "lost stolen card",
            "my card was stolen",
            "missing bank card",
            "transfer money pending",
            "bank transfer delayed",
            "money transfer missing",
        ],
        ["card", "card", "card", "transfer", "transfer", "transfer"],
    )

    monkeypatch.setattr(api, "load_model", lambda: model)

    with TestClient(api.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction(client):
    response = client.post(
        "/predict", json={"text": "lost stolen card"}
    )
    assert response.status_code == 200

    result = response.json()
    assert result["predicted_intent"] == "card"

    probabilities = [
        item["model_probability"]
        for item in result["top_predictions"]
    ]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(0 <= value <= 1 for value in probabilities)


@pytest.mark.parametrize("text", ["", "   ", "x" * 5001])
def test_invalid_text(client, text):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 422


def test_missing_text(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_unknown_vocabulary(client):
    response = client.post(
        "/predict", json={"text": "zzzxxyyqq"}
    )
    assert response.status_code == 200

    result = response.json()
    assert result["predicted_intent"] is None
    assert result["needs_human_review"] is True
    assert result["top_predictions"] == []
