"""
tests/test_frontend_workflow_simulation.py — Automated verification of all 6 frontend workflows
simulating the exact payload structure, query parameters, and data transformations of app.js.
"""

import uuid
import pytest
from fastapi.testclient import TestClient
from api.index import app


@pytest.fixture
def client():
    return TestClient(app)


def test_workflow_overview_page_data_flow(client):
    """Verifies that the /api/dashboard and /api/students endpoints return exact data needed by viewOverview()."""
    resp = client.get("/api/dashboard?student_id=student_001")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "total_sessions" in data["summary"]
    assert "topics_studied_count" in data["summary"]
    assert "average_understanding_score" in data["summary"]
    assert "average_actual_quiz_score" in data["summary"]
    assert "trend" in data
    assert "recent_sessions" in data
    assert "areas_of_confusion" in data
    assert "topic_wise_breakdown" in data

    # Verify that trend items contain fields expected by lineChart()
    for item in data.get("trend", []):
        assert "date" in item
        assert "topic" in item
        # Keys plotted by lineChart
        assert "understanding_score" in item
        assert "quiz_score" in item
        assert "overall_score" in item


def test_workflow_sessions_and_detail_data_flow(client):
    """Verifies session filtering, pagination, and full transcript/signal drilldown."""
    # List sessions
    resp = client.get("/api/sessions?student_id=student_001&limit=10&offset=0")
    assert resp.status_code == 200
    res_data = resp.json()
    assert "sessions" in res_data
    assert "total_count" in res_data

    if res_data["sessions"]:
        first_session = res_data["sessions"][0]
        session_id = first_session["session_id"]

        # Detail drill-down
        resp_det = client.get(f"/api/sessions/{session_id}?student_id=student_001")
        assert resp_det.status_code == 200
        det_data = resp_det.json()
        assert det_data["session_id"] == session_id
        assert "messages" in det_data
        assert "summary_metrics" in det_data
        assert "analytics" in det_data
        m = det_data["summary_metrics"]
        assert "predicted_class" in m
        assert "understanding_score" in m
        assert "expressed_confidence_level" in m


def test_workflow_ingest_dialog_and_pipeline(client):
    """Simulates openIngest() and submitIngest() from app.js."""
    uid = uuid.uuid4().hex[:8]
    student_id = f"student_sim_{uid}"

    # App.js parses text and creates messages list
    payload = {
        "student_id": student_id,
        "subject": "Distributed Systems",
        "topic": f"Raft Consensus {uid}",
        "source_agent": "claude",
        "messages": [
            {"role": "user", "content": f"How does leader election work in Raft consensus {uid}?"},
            {"role": "assistant", "content": "Raft uses randomized election timeouts to elect a leader with heartbeat intervals."},
            {"role": "user", "content": "I understand completely! Split votes are avoided by the randomized timeouts."},
        ],
    }

    resp = client.post("/api/ingest", json=payload)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["status"] == "success"
    session_id = res_data["session_id"]
    assert session_id is not None

    # Verify new session is listed in student sessions
    resp_list = client.get(f"/api/sessions?student_id={student_id}")
    assert resp_list.status_code == 200
    sessions = resp_list.json()["sessions"]
    assert any(s["session_id"] == session_id for s in sessions)


def test_workflow_quiz_dialog_and_deterministic_grading(client):
    """Simulates openQuiz() and submitQuiz() from app.js."""
    uid = uuid.uuid4().hex[:8]
    student_id = f"student_sim_{uid}"

    # 1. Ingest session
    ing_resp = client.post(
        "/api/ingest",
        json={
            "student_id": student_id,
            "subject": "Database Systems",
            "topic": f"Indexing {uid}",
            "text": "What is a B+ tree?\nIt is a self-balancing search tree used in database indices.\nUnderstood!",
        },
    )
    assert ing_resp.status_code == 200
    session_id = ing_resp.json()["session_id"]

    # 2. Submit quiz (matching app.js payload)
    quiz_payload = {
        "student_id": student_id,
        "session_id": session_id,
        "subject": "Database Systems",
        "topic": f"Indexing {uid}",
        "questions": [
            {
                "question_id": "q1",
                "question_text": "Where are values stored in a B+ tree?",
                "topic": f"Indexing {uid}",
                "student_answer": "leaf nodes",
                "correct_answer": "leaf nodes",
            },
            {
                "question_id": "q2",
                "question_text": "What is the height of a balanced tree with N keys?",
                "topic": f"Indexing {uid}",
                "student_answer": "O(N)",
                "correct_answer": "O(log N)",
            },
        ],
        "answer_key": {"q1": "leaf nodes", "q2": "O(log N)"},
        "student_answers": {"q1": "leaf nodes", "q2": "O(N)"},
    }

    resp = client.post("/api/quizzes", json=quiz_payload)
    assert resp.status_code == 200
    q_data = resp.json()
    assert q_data["total_questions"] == 2
    assert q_data["correct_count"] == 1
    assert q_data["incorrect_count"] == 1
    assert q_data["actual_score_percent"] == 50.0

    # 3. Verify session detail reflects the graded quiz and updated composite score
    s_detail = client.get(f"/api/sessions/{session_id}?student_id={student_id}").json()
    assert s_detail["summary_metrics"]["actual_quiz_score"] == 50.0
    assert s_detail["summary_metrics"]["overall_performance_score"] is not None
    assert s_detail["linked_quiz"] is not None
    assert s_detail["linked_quiz"]["actual_score_percent"] == 50.0


def test_workflow_progress_and_topic_detail(client):
    """Simulates viewProgress() and viewTopicDetail() from app.js."""
    resp_top = client.get("/api/topics")
    assert resp_top.status_code == 200
    topics_data = resp_top.json()
    assert "topics" in topics_data

    # Query student-performance
    resp_perf = client.get("/api/student-performance?student_id=student_001")
    assert resp_perf.status_code == 200
    perf_data = resp_perf.json()
    assert "total_sessions" in perf_data
    assert "topic_wise_breakdown" in perf_data
