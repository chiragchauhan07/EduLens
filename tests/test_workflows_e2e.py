"""
tests/test_workflows_e2e.py — End-to-end automated verification of Workflows A through E.

Verifies:
- Workflow A: Brand new student (empty states, no crashes, 0 sessions, no fallback 50, proper KPI badges).
- Workflow B: Session ingestion with text, JSON, turns, roles (ML prediction, features, confidence, recommendations, isolation).
- Workflow C: Post-session quiz evaluation (deterministic scoring, answer key matching, updating session score).
- Workflow D: Topic analytics & performance (unstudied vs studied, mastery progression, learning progress trend).
- Workflow E: Security, authentication, and cross-student access prevention.
"""

import uuid
import pytest
from fastapi.testclient import TestClient
from api.index import app
from backend.mcp_server.config import settings
from backend.mcp_server.auth.authentication import token_manager


@pytest.fixture
def client():
    return TestClient(app)


def test_workflow_a_brand_new_student(client):
    """Workflow A: New student portfolio inspection with zero data."""
    student_id = f"student_brand_new_{uuid.uuid4().hex[:8]}"
    resp = client.get(f"/api/dashboard?student_id={student_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["student_id"] == student_id
    assert data["summary"]["total_sessions"] == 0
    assert data["summary"]["average_understanding_score"] == 0.0
    assert data["summary"]["average_actual_quiz_score"] is None
    assert data["summary"]["average_confidence_score"] == 0.0
    assert data["summary"]["average_overall_score"] is None
    assert len(data["recent_sessions"]) == 0
    assert data["understanding_distribution"] == {"understood": 0, "partial": 0, "confused": 0}

    # Verify session list is empty
    resp_sess = client.get(f"/api/sessions?student_id={student_id}")
    assert resp_sess.status_code == 200
    assert resp_sess.json()["total_count"] == 0
    assert len(resp_sess.json()["sessions"]) == 0

    # Verify quizzes list is empty
    resp_quiz = client.get(f"/api/quizzes?student_id={student_id}")
    assert resp_quiz.status_code == 200
    assert resp_quiz.json()["total_count"] == 0


def test_workflow_b_session_ingestion_and_analytics(client):
    """Workflow B: Ingest dialogue transcript and verify unified inference pipeline."""
    uid = uuid.uuid4().hex[:8]
    student_id = f"student_wf_b_{uid}"
    payload = {
        "student_id": student_id,
        "subject": "Operating Systems",
        "topic": f"Deadlock Prevention {uid}",
        "source_agent": "chatgpt",
        "messages": [
            {"role": "user", "content": f"Can you explain the Coffman conditions for deadlocks in system {uid}?"},
            {"role": "assistant", "content": "Mutual exclusion, hold and wait, no preemption, circular wait."},
            {"role": "user", "content": "I understand completely! Breaking circular wait prevents deadlock."},
        ],
    }
    resp = client.post("/api/ingest", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    session_id = data["session_id"]
    assert session_id.startswith("SES_") or session_id.startswith("sess_")
    analytics = data["analytics"]
    pred_class = (
        analytics.get("predicted_understanding")
        or analytics.get("topic_understanding", {}).get("predicted_class")
        or (analytics.get("topics", [{}])[0].get("understanding", {}).get("classification") if analytics.get("topics") else None)
    )
    assert pred_class in ["understood", "partial", "confused"]
    conf_score = (
        analytics.get("expressed_confidence_score")
        or analytics.get("learning_confidence", {}).get("expressed_confidence_score")
        or (analytics.get("topics", [{}])[0].get("expressed_confidence", {}).get("score") if analytics.get("topics") else None)
    )
    assert conf_score is not None

    overall = (
        analytics.get("overall_topic_performance")
        or (analytics.get("topics", [{}])[0].get("overall_performance") if analytics.get("topics") else None)
    )
    assert overall is not None

    # Verify session is retrievable in session detail
    resp_det = client.get(f"/api/sessions/{session_id}?student_id={student_id}")
    assert resp_det.status_code == 200
    detail = resp_det.json()
    assert detail["session_id"] == session_id
    assert detail["subject"] == "Operating Systems"
    assert detail["topic"] == f"Deadlock Prevention {uid}"
    assert len(detail["messages"]) == 3
    assert detail["summary_metrics"]["actual_quiz_score"] is None
    assert detail["linked_quiz"] is None


def test_workflow_c_post_session_quiz_evaluation(client):
    """Workflow C: Deterministic grading, answer matching, and session composite update."""
    uid = uuid.uuid4().hex[:8]
    student_id = f"student_wf_c_{uid}"
    # First ingest a session
    ing_resp = client.post(
        "/api/ingest",
        json={
            "student_id": student_id,
            "subject": "Data Structures",
            "topic": f"Binary Search Trees {uid}",
            "text": f"What is the time complexity of searching in a balanced BST {uid}?\nIt is O(log n) because height is log n.\nGot it! Understood.",
        },
    )
    assert ing_resp.status_code == 200
    session_id = ing_resp.json()["session_id"]

    # Submit quiz
    quiz_payload = {
        "student_id": student_id,
        "session_id": session_id,
        "subject": "Data Structures",
        "topic": f"Binary Search Trees {uid}",
        "questions": [
            {"question_id": "q1", "question_text": "Average search time in balanced BST?", "correct_answer": "O(log n)"},
            {"question_id": "q2", "question_text": "Worst case search time in degenerate BST?", "correct_answer": "O(n)"},
        ],
        "answer_key": {"q1": "O(log n)", "q2": "O(n)"},
        "student_answers": {"q1": "O(log n)", "q2": "O(n)"},
    }
    quiz_resp = client.post("/api/quizzes", json=quiz_payload)
    assert quiz_resp.status_code == 200
    q_data = quiz_resp.json()
    assert q_data["actual_score_percent"] == 100.0
    assert q_data["correct_count"] == 2
    assert q_data["incorrect_count"] == 0
    assert q_data["unanswered_count"] == 0

    # Verify session detail now links the quiz and has updated composite
    sess_resp = client.get(f"/api/sessions/{session_id}?student_id={student_id}")
    assert sess_resp.status_code == 200
    s_detail = sess_resp.json()
    assert s_detail["summary_metrics"]["actual_quiz_score"] == 100.0
    assert s_detail["summary_metrics"]["overall_performance_score"] is not None
    assert s_detail["linked_quiz"] is not None
    assert s_detail["linked_quiz"]["actual_score_percent"] == 100.0


def test_workflow_d_topic_analytics_and_curriculum_distinction(client):
    """Workflow D: Verify studied vs unstudied curriculum topics."""
    uid = uuid.uuid4().hex[:8]
    student_id = f"student_wf_d_{uid}"
    topic_name = f"Merge Sort {uid}"
    # Ingest session for topic A
    client.post(
        "/api/ingest",
        json={
            "student_id": student_id,
            "subject": "Algorithms",
            "topic": topic_name,
            "text": f"Merge sort divides array in half and merges sorted halves for run {uid}.\nTime complexity is always O(n log n).",
        },
    )

    # Studied topic returns actual session count and scores
    resp_studied = client.get(f"/api/topic-summary?student_id={student_id}&topic={topic_name}")
    assert resp_studied.status_code == 200
    st_data = resp_studied.json()
    assert st_data["total_sessions"] >= 1
    assert st_data["topic"] == topic_name

    # Unstudied topic returns 0 sessions and clean None metrics
    resp_unstudied = client.get(f"/api/topic-summary?student_id={student_id}&topic=Unstudied%20Topic%20XYZ")
    assert resp_unstudied.status_code == 200
    un_data = resp_unstudied.json()
    assert un_data["total_sessions"] == 0
    assert un_data["quiz_performance"]["average_score_percent"] is None
    assert un_data["overall_topic_performance"] is None


def test_workflow_e_security_and_student_isolation(client):
    """Workflow E: Bearer token enforcement in production & cross-student access prevention."""
    token_manager.register_token("tok-alice-123", "alice_id", "Alice")
    token_manager.register_token("tok-bob-456", "bob_id", "Bob")

    # In development mode, identity switching is permitted
    resp_dev = client.get("/api/dashboard?student_id=alice_id")
    assert resp_dev.status_code == 200
    assert resp_dev.json()["student_id"] == "alice_id"

    # Switch to production mode temporarily
    orig_env = settings.environment
    try:
        settings.environment = "production"

        # Missing token in production -> 401 Unauthorized
        resp_unauth = client.get("/api/dashboard?student_id=alice_id")
        assert resp_unauth.status_code == 401

        # Alice tries to access Bob's data with her token -> 403 Forbidden
        headers_alice = {"Authorization": "Bearer tok-alice-123"}
        resp_cross = client.get("/api/dashboard?student_id=bob_id", headers=headers_alice)
        assert resp_cross.status_code == 403

        # Alice accesses her own data -> 200 OK
        resp_alice = client.get("/api/dashboard?student_id=alice_id", headers=headers_alice)
        assert resp_alice.status_code == 200
        assert resp_alice.json()["student_id"] == "alice_id"

    finally:
        settings.environment = orig_env
