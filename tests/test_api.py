"""
test_api.py — API integration tests for EduLens.

Tests:
- GET  /api/health
- GET  /api/model-info
- GET  /api/sample-conversation
- POST /api/analyze (JSON body)
- POST /api/analyze (Multipart file upload)
- POST /api/analyze (Error handling: invalid JSON, empty file, non-json extension)
- GET  / (Frontend static HTML)
"""

import io
import json
import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_api_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "EduLens"
    assert data["model_loaded"] is True


def test_api_model_info_endpoint():
    resp = client.get("/api/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "Logistic Regression (Balanced)"
    assert data["features_dimension"] == 108
    assert "Understood" in data["supported_understanding_classes"]
    assert "Partially Understood" in data["supported_understanding_classes"]
    assert "Confused" in data["supported_understanding_classes"]
    assert data["status"] == "operational"


def test_api_sample_conversation_endpoint():
    resp = client.get("/api/sample-conversation")
    assert resp.status_code == 200
    data = resp.json()
    assert "topic" in data
    assert "messages" in data
    assert len(data["messages"]) > 0


def test_api_analyze_with_json_body():
    payload = {
        "topic": "Newton's Second Law",
        "messages": [
            {"role": "assistant", "content": "What is the relationship between force, mass, and acceleration?"},
            {"role": "student", "content": "can we review Newton's laws?"},
            {"role": "student", "content": "so basically force equals mass times acceleration F=ma?"},
            {"role": "student", "content": "got it, that makes complete sense! Thanks!"},
        ],
    }
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["topic"] == "Newton's Second Law"
    assert data["understanding"]["label"] == "Understood"
    assert "class_probabilities" in data["understanding"]
    assert data["expressed_confidence"]["label"] in ["High Confidence", "Moderate Confidence", "Low Confidence"]
    assert data["conversation_stats"]["student_messages"] == 3


def test_api_analyze_with_multipart_file_upload():
    content = json.dumps({
        "topic": "Graph Algorithms",
        "messages": [
            {"role": "student", "content": "i don't understand Dijkstra's algorithm, it's so confusing"},
            {"role": "student", "content": "why does it fail with negative cycles? I'm completely lost"},
        ],
    }).encode("utf-8")

    files = {"file": ("graph_session.json", io.BytesIO(content), "application/json")}
    resp = client.post("/api/analyze", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["topic"] == "Graph Algorithms"
    assert data["understanding"]["label"] == "Confused"
    assert data["conversation_stats"]["student_messages"] == 2


def test_api_analyze_unsupported_file_extension():
    files = {"file": ("notes.txt", io.BytesIO(b"random text"), "text/plain")}
    resp = client.post("/api/analyze", files=files)
    assert resp.status_code == 400
    assert "Unsupported file format" in resp.json()["detail"]


def test_api_analyze_empty_file():
    files = {"file": ("empty.json", io.BytesIO(b""), "application/json")}
    resp = client.post("/api/analyze", files=files)
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"]


def test_api_analyze_invalid_json():
    files = {"file": ("broken.json", io.BytesIO(b"{bad json syntax"), "application/json")}
    resp = client.post("/api/analyze", files=files)
    assert resp.status_code == 400
    assert "Invalid JSON" in resp.json()["detail"]


def test_api_analyze_missing_student_messages():
    payload = {
        "topic": "Empty Student Chat",
        "messages": [
            {"role": "assistant", "content": "Hello, is anyone there?"},
        ],
    }
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 400
    assert "only assistant/tutor messages" in resp.json()["detail"]


def test_frontend_home_served():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "EduLens" in resp.text
    assert "Student Learning Intelligence" in resp.text
