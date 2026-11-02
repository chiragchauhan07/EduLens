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
