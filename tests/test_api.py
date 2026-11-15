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


def test_api_predict_with_explain():
    payload = {
        "text": "I really don't understand how recursion works. I am lost and confused.",
        "explain": True,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "feature_explanation" in data
    assert data["feature_explanation"] is not None
    assert "top_positive_features" in data["feature_explanation"]
    assert "top_negative_features" in data["feature_explanation"]


def test_api_explainability_endpoint():
    res = client.get("/api/explainability")
    assert res.status_code == 200
    data = res.json()
    assert "logistic_regression" in data
    assert "linear_svm" in data
    assert "naive_bayes" in data
    assert "quiz_regressor" in data


def test_api_students():
    res = client.get("/api/students")
    assert res.status_code == 200
    data = res.json()
    assert "students" in data
    assert "student_001" in data["students"]


def test_api_dashboard():
    res = client.get("/api/dashboard?student_id=student_001")
    assert res.status_code == 200
    data = res.json()
    assert data["student_id"] == "student_001"
    assert "summary" in data
    assert "understanding_distribution" in data
    assert "recent_sessions" in data
    assert data["summary"]["total_sessions"] >= 0


def test_api_dashboard_empty_student():
    res = client.get("/api/dashboard?student_id=student_unseen_999")
    assert res.status_code == 200
    data = res.json()
    assert data["summary"]["total_sessions"] == 0
    assert data["summary"]["topics_studied_count"] == 0
    assert data["understanding_distribution"] == {"understood": 0, "partial": 0, "confused": 0}
    assert len(data["recent_sessions"]) == 0


def test_api_sessions_list_and_search():
    res = client.get("/api/sessions?student_id=student_001&limit=5")
    assert res.status_code == 200
    data = res.json()
    assert "total_count" in data
    assert "sessions" in data

    # Test search
    res_search = client.get("/api/sessions?student_id=student_001&search=Binary")
    assert res_search.status_code == 200
    assert "sessions" in res_search.json()


def test_api_quizzes_list_and_submit():
    # 1. List quizzes
    res_list = client.get("/api/quizzes?student_id=student_001")
    assert res_list.status_code == 200
    data_list = res_list.json()
    assert "total_count" in data_list
    assert "quizzes" in data_list

    # 2. Submit a quiz
    payload = {
        "student_id": "student_001",
        "subject": "Computer Science",
        "topic": "Binary Search",
        "questions": [
            {"question_id": "q1", "student_answer": "O(log n)", "correct_answer": "O(log n)"},
            {"question_id": "q2", "student_answer": "sorted", "correct_answer": "sorted"},
        ]
    }
    res_sub = client.post("/api/quizzes", json=payload)
    assert res_sub.status_code == 200
    data_sub = res_sub.json()
    assert data_sub["actual_score_percent"] == 100.0
    assert data_sub["correct_count"] == 2
    assert "quiz_id" in data_sub

    # 3. Retrieve specific quiz
    quiz_id = data_sub["quiz_id"]
    res_get = client.get(f"/api/quizzes/{quiz_id}?student_id=student_001")
    assert res_get.status_code == 200
    data_get = res_get.json()
    assert data_get["quiz_id"] == quiz_id
    assert data_get["total_questions"] == 2


def test_api_ingest_and_session_detail():
    payload = {
        "student_id": "student_001",
        "subject": "Data Structures",
        "topic": "Hash Tables",
        "source_agent": "chatgpt",
        "messages": [
            {"role": "user", "content": "How does quadratic probing resolve collisions in a hash table?"},
            {"role": "assistant", "content": "Quadratic probing uses an open addressing scheme with polynomial steps h(k, i) = (h'(k) + c1*i + c2*i^2) mod m."},
            {"role": "user", "content": "I understand completely! That prevents primary clustering that happens with linear probing. Thank you!"}
        ]
    }
    res_ingest = client.post("/api/ingest", json=payload)
    assert res_ingest.status_code == 200
    data_ingest = res_ingest.json()
    assert "session_id" in data_ingest
    assert data_ingest["status"] in ("success", "duplicate_detected")

    session_id = data_ingest["session_id"]
    res_detail = client.get(f"/api/sessions/{session_id}?student_id=student_001")
    assert res_detail.status_code == 200
    data_detail = res_detail.json()
    assert data_detail["session_id"] == session_id
    assert len(data_detail["messages"]) >= 3
    assert data_detail["analytics"] is not None
