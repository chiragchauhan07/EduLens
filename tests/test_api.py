"""
test_api.py — Automated tests for Vercel Serverless API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.index import app

client = TestClient(app)


def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_api_samples():
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.json()
    assert "samples" in data
    assert len(data["samples"]) >= 3


def test_api_topics():
    res = client.get("/api/topics")
    assert res.status_code == 200
    data = res.json()
    assert "topics" in data
    assert len(data["topics"]) > 0


def test_api_predict():
    payload = {
        "text": "I really don't understand how recursion works. I am lost and confused."
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_class" in data
    assert data["predicted_class"] in {"confused", "partial", "understood"}
    assert "probabilities" in data
    assert "confidence" in data
    assert "conversational_signals" in data
    assert "predicted_quiz_score" in data


def test_api_predict_with_algorithm():
    for algo in ["logistic_regression", "naive_bayes", "linear_svm"]:
        payload = {
            "conversation": [
                "explain binary search tree insertion",
                "got it, smaller goes left and larger goes right! makes complete sense now.",
            ],
            "algorithm": algo,
        }
        res = client.post("/api/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["predicted_class"] in {"confused", "partial", "understood"}


def test_api_model_info():
    res = client.get("/api/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "model_name" in data
    assert "evaluation_metrics" in data


def test_api_model_comparison():
    res = client.get("/api/model-comparison")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert "logistic_regression" in data["models"]
    assert "naive_bayes" in data["models"]
    assert "linear_svm" in data["models"]


def test_api_quiz_model_info():
    res = client.get("/api/quiz-model-info")
    assert res.status_code == 200
    data = res.json()
    assert "task" in data
    assert "models" in data


def test_api_predict_empty_fails():
    res = client.post("/api/predict", json={})
    assert res.status_code == 400
