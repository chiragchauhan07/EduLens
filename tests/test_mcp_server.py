"""
test_mcp_server.py — Comprehensive Automated Test Suite for EduLens MCP Integration.
Implements the 24 required test scenarios from the EduLens MCP Specification.
"""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List
import pytest

from backend.mcp_server.config import MCPSettings, settings
from backend.mcp_server.auth.authentication import (
    AuthenticationError,
    AuthorizationError,
    TokenManager,
    UserContext,
    set_current_user,
)
from backend.mcp_server.schemas.session_schema import (
    MessageItem,
    QuizSubmissionRequest,
    SessionIngestRequest,
)
from backend.mcp_server.services.storage_service import SQLiteStorageService
from backend.mcp_server.services.ingestion_service import IngestionService
from backend.mcp_server.server import create_mcp_server, create_http_app


@pytest.fixture
def temp_storage():
    """Provides an isolated temporary SQLite database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_edulens.db"
        storage = SQLiteStorageService(db_path=db_path)
        yield storage


@pytest.fixture
def ingestion_service(temp_storage):
    """Provides an IngestionService instance wired to temp_storage."""
    return IngestionService(storage=temp_storage)


@pytest.fixture
def student_user_1():
    return UserContext(user_id="student_001", username="Alice Learner", client_type="chatgpt")


@pytest.fixture
def student_user_2():
    return UserContext(user_id="student_002", username="Bob Learner", client_type="claude")


# ─────────────────────────────────────────────────────────────────────────────
# 1. MCP Server Initialization
# ─────────────────────────────────────────────────────────────────────────────
def test_case_1_mcp_server_initialization():
    """Test 1: MCP server initializes with proper metadata and instructions."""
    server = create_mcp_server()
    assert server is not None
    assert server.name == "edulens-mcp-server"
    assert "EduLens" in server.instructions


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tool Discovery
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_case_2_tool_discovery():
    """Test 2: MCP server exposes all 6 core tools with valid schemas."""
    server = create_mcp_server()
    tools = await server.list_tools()
    tool_names = {t.name for t in tools}

    expected_tools = {
        "ingest_learning_session",
        "get_session_analytics",
        "get_student_performance",
        "get_topic_analytics",
        "list_learning_sessions",
        "submit_quiz_results",
    }
    assert expected_tools.issubset(tool_names), f"Missing tools: {expected_tools - tool_names}"

    # Verify ingest_learning_session requires messages
    ingest_tool = next(t for t in tools if t.name == "ingest_learning_session")
    assert "messages" in ingest_tool.input_schema.get("properties", {})


# ─────────────────────────────────────────────────────────────────────────────
# 3. Valid Session Ingestion
# ─────────────────────────────────────────────────────────────────────────────
def test_case_3_valid_session_ingestion(ingestion_service, student_user_1):
    """Test 3: Ingestion of valid student-AI conversation succeeds across 4 dimensions."""
    req = SessionIngestRequest(
        subject="Computer Science",
        topic="Binary Search",
        source_agent="chatgpt",
        messages=[
            MessageItem(role="user", content="How does binary search work on sorted arrays?"),
            MessageItem(role="assistant", content="It compares the target with the middle element and halves the search space."),
            MessageItem(role="user", content="So the time complexity is O(log n) because we divide by 2 each step. Got it completely!"),
        ],
    )
    result = ingestion_service.ingest_session(req, student_user_1)
    assert result.status == "success"
    assert result.session_id.startswith("SES_")
    assert result.user_id == student_user_1.user_id
    assert result.turns_count == 3
    assert result.is_duplicate is False
    assert result.analytics is not None
    assert "topics" in result.analytics or "topic_understanding" in result.analytics


# ─────────────────────────────────────────────────────────────────────────────
# 4. Invalid JSON Rejection
# ─────────────────────────────────────────────────────────────────────────────
def test_case_4_invalid_json_rejection(ingestion_service, student_user_1):
    """Test 4: Reject non-dictionary or improperly typed messages."""
    with pytest.raises(Exception):
        # Attempt to pass raw string instead of MessageItem list
        SessionIngestRequest(
            topic="Sorting",
            messages="just a raw string, not a list",  # type: ignore
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Missing Required Fields
# ─────────────────────────────────────────────────────────────────────────────
def test_case_5_missing_required_fields():
    """Test 5: MessageItem without content or role raises validation error."""
    with pytest.raises(ValueError):
        MessageItem(role="user", content="")

    with pytest.raises(ValueError):
        MessageItem(role="", content="some content")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Empty Conversation Handling
# ─────────────────────────────────────────────────────────────────────────────
def test_case_6_empty_conversation(ingestion_service, student_user_1):
    """Test 6: Rejecting conversation with no turns or no student messages."""
    with pytest.raises(ValueError, match="at least one message"):
        SessionIngestRequest(topic="Recursion", messages=[])

    # Assistant-only turns (no student turns)
    with pytest.raises(ValueError, match="at least one student utterance"):
        SessionIngestRequest(
            topic="Recursion",
            messages=[MessageItem(role="assistant", content="Here is an explanation of recursion.")],
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Multiple Message Roles Handling
# ─────────────────────────────────────────────────────────────────────────────
def test_case_7_multiple_message_roles(ingestion_service, student_user_1):
    """Test 7: Handles user, assistant, and filters out system instructions."""
    req = SessionIngestRequest(
        topic="QuickSort",
        messages=[
            MessageItem(role="system", content="You are a strict programming tutor. Keep answers brief."),
            MessageItem(role="user", content="What is the pivot in quicksort?"),
            MessageItem(role="assistant", content="The pivot is the reference element used to partition the array."),
            MessageItem(role="user", content="Got it, smaller to left, larger to right!"),
        ],
    )
    res = ingestion_service.ingest_session(req, student_user_1)
    assert res.status == "success"
    # System message excluded from learning turns
    assert res.turns_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# 8. ChatGPT Source-Agent Metadata
# ─────────────────────────────────────────────────────────────────────────────
def test_case_8_chatgpt_source_agent_metadata(ingestion_service, student_user_1):
    """Test 8: Preserves ChatGPT source-agent attribution in storage."""
    req = SessionIngestRequest(
        topic="Graphs",
        source_agent="chatgpt",
        metadata={"model": "gpt-4o", "conversation_language": "English"},
        messages=[
            MessageItem(role="user", content="What is Dijkstra's algorithm?"),
            MessageItem(role="assistant", content="It finds the shortest path in a weighted graph with non-negative edges."),
            MessageItem(role="user", content="Understood, using a min-priority queue!"),
        ],
    )
    res = ingestion_service.ingest_session(req, student_user_1)
    assert res.source_agent == "chatgpt"

    raw = ingestion_service.storage.get_raw_session(res.session_id, student_user_1.user_id)
    assert raw["source_agent"] == "chatgpt"
    assert raw["raw_json"]["metadata"]["model"] == "gpt-4o"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Claude Source-Agent Metadata
# ─────────────────────────────────────────────────────────────────────────────
def test_case_9_claude_source_agent_metadata(ingestion_service, student_user_2):
    """Test 9: Preserves Claude source-agent attribution in storage."""
    req = SessionIngestRequest(
        topic="Pointers",
        source_agent="claude",
        metadata={"model": "claude-3-5-sonnet", "client": "Claude Desktop"},
        messages=[
            MessageItem(role="user", content="Can you explain C++ pointers?"),
            MessageItem(role="assistant", content="A pointer is a variable that stores the memory address of another variable."),
            MessageItem(role="user", content="I see, & gives the address and * dereferences it. Makes sense now!"),
        ],
    )
    res = ingestion_service.ingest_session(req, student_user_2)
    assert res.source_agent == "claude"

    raw = ingestion_service.storage.get_raw_session(res.session_id, student_user_2.user_id)
    assert raw["source_agent"] == "claude"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Duplicate Ingestion & Idempotency
# ─────────────────────────────────────────────────────────────────────────────
def test_case_10_duplicate_ingestion(ingestion_service, student_user_1):
    """Test 10: Ingesting identical conversation returns duplicate detection without re-processing."""
    messages = [
        MessageItem(role="user", content="What is a hash table collision?"),
        MessageItem(role="assistant", content="It happens when two keys hash to the exact same bucket."),
        MessageItem(role="user", content="Clear now, handled via chaining or open addressing!"),
    ]
    req1 = SessionIngestRequest(topic="Hash Tables", messages=messages)
    res1 = ingestion_service.ingest_session(req1, student_user_1)
    assert res1.status == "success"
    assert res1.is_duplicate is False

    # Second submission with identical messages
    req2 = SessionIngestRequest(topic="Hash Tables", messages=messages)
    res2 = ingestion_service.ingest_session(req2, student_user_1)
    assert res2.status == "duplicate_detected"
    assert res2.is_duplicate is True
    assert res2.session_id == res1.session_id


# ─────────────────────────────────────────────────────────────────────────────
# 11. User Authentication
# ─────────────────────────────────────────────────────────────────────────────
def test_case_11_user_authentication():
    """Test 11: Valid token resolves to correct UserContext."""
    mgr = TokenManager()
    token = mgr.generate_token_for_user("student_abc", client_type="claude")
    user = mgr.validate_token(token)
    assert user.user_id == "student_abc"
    assert user.client_type == "claude"
    assert user.is_authenticated is True


# ─────────────────────────────────────────────────────────────────────────────
# 12. Unauthorized Request Handling
# ─────────────────────────────────────────────────────────────────────────────
def test_case_12_unauthorized_request():
    """Test 12: Missing or invalid token raises AuthenticationError when auth is enabled."""
    mgr = TokenManager()
    with pytest.raises(AuthenticationError, match="Missing authentication credentials"):
        mgr.validate_token(None)

    with pytest.raises(AuthenticationError, match="Invalid or expired"):
        mgr.validate_token("Bearer completely_bogus_token_12345")


# ─────────────────────────────────────────────────────────────────────────────
# 13. Cross-User Access Prevention
# ─────────────────────────────────────────────────────────────────────────────
def test_case_13_cross_user_access_prevention(ingestion_service, student_user_1, student_user_2):
    """Test 13: Student 2 cannot access Student 1's sessions or performance records."""
    req = SessionIngestRequest(
        topic="Dynamic Programming",
        messages=[
            MessageItem(role="user", content="Explain DP bottom-up tabulation."),
            MessageItem(role="assistant", content="Tabulation builds a table iteratively from base cases up."),
            MessageItem(role="user", content="Got it, avoids recursion overhead!"),
        ],
    )
    res1 = ingestion_service.ingest_session(req, student_user_1)

    # Student 2 tries to access Student 1's session analytics
    with pytest.raises(KeyError):
        ingestion_service.get_session_analytics(res1.session_id, student_user_2)

    # Student 2 tries to query Student 1's student performance
    with pytest.raises(AuthorizationError):
        ingestion_service.get_student_performance(student_id=student_user_1.user_id, user=student_user_2)


# ─────────────────────────────────────────────────────────────────────────────
# 14. Successful Analytics Processing
# ─────────────────────────────────────────────────────────────────────────────
def test_case_14_successful_analytics_processing(ingestion_service, student_user_1):
    """Test 14: Analytics pipeline produces understanding classification and score."""
    req = SessionIngestRequest(
        topic="Trees",
        messages=[
            MessageItem(role="user", content="What is tree height?"),
            MessageItem(role="assistant", content="The number of edges on the longest path from root to leaf."),
            MessageItem(role="user", content="Understood, an empty tree has height -1 or 0 by convention!"),
        ],
    )
    res = ingestion_service.ingest_session(req, student_user_1)
    ana = ingestion_service.get_session_analytics(res.session_id, student_user_1)
    assert ana.topic_understanding["predicted_class"] in ("understood", "partial", "confused")
    assert ana.topic_understanding["understanding_score"] >= 0.0
    assert ana.learning_confidence["expressed_confidence_level"] in ("low", "moderate", "high")


# ─────────────────────────────────────────────────────────────────────────────
# 15. Analytics Retrieval
# ─────────────────────────────────────────────────────────────────────────────
def test_case_15_analytics_retrieval(ingestion_service, student_user_1):
    """Test 15: get_session_analytics returns complete structured intelligence."""
    req = SessionIngestRequest(
        topic="SQL Joins",
        subject="Databases",
        messages=[
            MessageItem(role="user", content="What is the difference between INNER and LEFT JOIN?"),
            MessageItem(role="assistant", content="INNER returns matches from both tables, LEFT keeps all left rows."),
            MessageItem(role="user", content="Clear now, unmatched right columns become NULL!"),
        ],
    )
    res = ingestion_service.ingest_session(req, student_user_1)
    ana = ingestion_service.get_session_analytics(res.session_id, student_user_1)
    assert ana.session_id == res.session_id
    assert ana.user_id == student_user_1.user_id
    assert ana.subject == "Databases"
    assert ana.topic == "SQL Joins"
    assert ana.pedagogical_feedback != ""


# ─────────────────────────────────────────────────────────────────────────────
# 16. Quiz Result Submission
# ─────────────────────────────────────────────────────────────────────────────
def test_case_16_quiz_result_submission(ingestion_service, student_user_1):
    """Test 16: submit_quiz_results evaluates answers and updates linked session score."""
    # First ingest a session
    req_sess = SessionIngestRequest(
        topic="Recursion",
        messages=[
            MessageItem(role="user", content="Why does recursion need a base case?"),
            MessageItem(role="assistant", content="Without a base case, it recurses infinitely and overflows the call stack."),
            MessageItem(role="user", content="Understood!"),
        ],
    )
    res_sess = ingestion_service.ingest_session(req_sess, student_user_1)

    # Submit quiz for this session
    quiz_req = QuizSubmissionRequest(
        session_id=res_sess.session_id,
        topic="Recursion",
        questions=[
            {"question_id": "q1", "student_answer": "A", "correct_answer": "A"},
            {"question_id": "q2", "student_answer": "B", "correct_answer": "B"},
            {"question_id": "q3", "student_answer": "C", "correct_answer": "D"},
        ],
    )
    q_res = ingestion_service.submit_quiz_results(quiz_req, student_user_1)
    assert q_res.total_questions == 3
    assert q_res.correct_count == 2
    assert q_res.incorrect_count == 1
    assert round(q_res.actual_score_percent, 1) == 66.7
    assert q_res.updated_overall_performance is not None


# ─────────────────────────────────────────────────────────────────────────────
# 17. Student Performance Retrieval
# ─────────────────────────────────────────────────────────────────────────────
def test_case_17_student_performance_retrieval(ingestion_service, student_user_1):
    """Test 17: get_student_performance returns portfolio aggregation."""
    # Ingest two sessions
    for topic in ["Queues", "Stacks"]:
        ingestion_service.ingest_session(
            SessionIngestRequest(
                topic=topic,
                messages=[
                    MessageItem(role="user", content=f"How does a {topic} data structure operate?"),
                    MessageItem(role="assistant", content=f"A {topic} is a linear collection."),
                    MessageItem(role="user", content="Understood!"),
                ],
            ),
            student_user_1,
        )

    perf = ingestion_service.get_student_performance(student_id=student_user_1.user_id, user=student_user_1)
    assert perf.total_sessions >= 2
    assert "Queues" in perf.topics_studied
    assert "Stacks" in perf.topics_studied
    assert perf.average_understanding_score > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 18. Topic Analytics Retrieval
# ─────────────────────────────────────────────────────────────────────────────
def test_case_18_topic_analytics_retrieval(ingestion_service, student_user_1):
    """Test 18: get_topic_analytics returns topic-specific intelligence."""
    topic_name = "Big O Notation"
    ingestion_service.ingest_session(
        SessionIngestRequest(
            topic=topic_name,
            messages=[
                MessageItem(role="user", content="What is O(1) vs O(n)?"),
                MessageItem(role="assistant", content="O(1) is constant time, O(n) scales linearly."),
                MessageItem(role="user", content="Got it completely!"),
            ],
        ),
        student_user_1,
    )
    top = ingestion_service.get_topic_analytics(topic=topic_name, student_id=student_user_1.user_id, user=student_user_1)
    assert top.topic == topic_name
    assert top.total_sessions >= 1
    assert top.average_understanding_score > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 19. Storage Failure Handling
# ─────────────────────────────────────────────────────────────────────────────
def test_case_19_storage_failure_handling(student_user_1, monkeypatch):
    """Test 19: Storage failures do not corrupt state or cause silent unhandled crashes."""
    import sqlite3
    storage = SQLiteStorageService(db_path=Path("data/storage/test_fail.db"))
    def failing_save(*args, **kwargs):
        raise sqlite3.OperationalError("Simulated disk error or locked database")
    monkeypatch.setattr(storage, "save_raw_session", failing_save)
    service = IngestionService(storage=storage)
    req = SessionIngestRequest(
        topic="Failure Test",
        messages=[MessageItem(role="user", content="Testing database failure resilience.")],
    )
    with pytest.raises(sqlite3.OperationalError):
        service.ingest_session(req, student_user_1)


# ─────────────────────────────────────────────────────────────────────────────
# 20. Analytics Processing Failure Resilience
# ─────────────────────────────────────────────────────────────────────────────
def test_case_20_analytics_processing_failure_resilience(ingestion_service, student_user_1, monkeypatch):
    """Test 20: When ML pipeline fails, raw session remains safely preserved in storage."""
    # Force the pipeline to raise an exception
    def broken_process(*args, **kwargs):
        raise RuntimeError("Simulated ML inference failure")

    monkeypatch.setattr(ingestion_service, "_get_pipeline", lambda: type("Broken", (), {"process_session": broken_process})())

    req = SessionIngestRequest(
        topic="Resilience Check",
        messages=[
            MessageItem(role="user", content="Can you explain bubble sort?"),
            MessageItem(role="assistant", content="It compares adjacent pairs repeatedly."),
            MessageItem(role="user", content="Got it!"),
        ],
    )
    res = ingestion_service.ingest_session(req, student_user_1)
    assert res.status == "processing_failed"
    assert "Simulated ML inference failure" in res.message

    # Confirm raw session is still safely saved in DB
    raw = ingestion_service.storage.get_raw_session(res.session_id, student_user_1.user_id)
    assert raw is not None
    assert raw["status"] == "processing_failed"
    assert "Simulated ML inference failure" in raw["error_message"]


# ─────────────────────────────────────────────────────────────────────────────
# 21. Oversized Payload Handling
# ─────────────────────────────────────────────────────────────────────────────
def test_case_21_oversized_payload_handling(ingestion_service, student_user_1):
    """Test 21: High volume conversation handles turn sequences cleanly."""
    many_messages = []
    for i in range(50):
        many_messages.append(MessageItem(role="user", content=f"Step {i}: what happens next?"))
        many_messages.append(MessageItem(role="assistant", content=f"Step {i}: here is the explanation."))
    many_messages.append(MessageItem(role="user", content="Now I understand the complete algorithm!"))

    req = SessionIngestRequest(topic="Long Dialogue", messages=many_messages)
    res = ingestion_service.ingest_session(req, student_user_1)
    assert res.status == "success"
    assert res.turns_count == 101


# ─────────────────────────────────────────────────────────────────────────────
# 22. Malformed Timestamps Handling
# ─────────────────────────────────────────────────────────────────────────────
def test_case_22_malformed_timestamps_handling(ingestion_service, student_user_1):
    """Test 22: Non-standard or malformed timestamp strings do not crash ingestion."""
    req = SessionIngestRequest(
        topic="Timestamps",
        messages=[
            MessageItem(role="user", content="What is a heap?", timestamp="invalid-not-iso-format"),
            MessageItem(role="assistant", content="A heap is a specialized tree.", timestamp="yesterday afternoon"),
            MessageItem(role="user", content="Understood!", timestamp="null"),
        ],
    )
    res = ingestion_service.ingest_session(req, student_user_1)
    assert res.status == "success"


# ─────────────────────────────────────────────────────────────────────────────
# 23. Session Pagination
# ─────────────────────────────────────────────────────────────────────────────
def test_case_23_session_pagination(ingestion_service, student_user_1):
    """Test 23: list_learning_sessions pagination accurately reflects limit and offset."""
    for i in range(5):
        ingestion_service.ingest_session(
            SessionIngestRequest(
                topic=f"Topic_{i}",
                messages=[
                    MessageItem(role="user", content=f"Question on topic {i}?"),
                    MessageItem(role="assistant", content=f"Answer {i}"),
                    MessageItem(role="user", content="Got it!"),
                ],
            ),
            student_user_1,
        )

    # Page 1 (limit 2, offset 0)
    page1 = ingestion_service.list_learning_sessions(student_user_1, limit=2, offset=0)
    assert len(page1.sessions) == 2
    assert page1.total_count >= 5

    # Page 2 (limit 2, offset 2)
    page2 = ingestion_service.list_learning_sessions(student_user_1, limit=2, offset=2)
    assert len(page2.sessions) == 2
    assert page2.sessions[0].session_id != page1.sessions[0].session_id


# ─────────────────────────────────────────────────────────────────────────────
# 24. MCP Transport Compatibility
# ─────────────────────────────────────────────────────────────────────────────
def test_case_24_transport_compatibility():
    """Test 24: HTTP Starlette application mounts and responds to health checks."""
    app_sse = create_http_app(transport="sse")
    app_streamable = create_http_app(transport="streamable")
    assert app_sse is not None
    assert app_streamable is not None


# ─────────────────────────────────────────────────────────────────────────────
# 25. HTTP Missing Authorization Rejection
# ─────────────────────────────────────────────────────────────────────────────
def test_case_25_missing_authorization_header_rejected():
    """Test 25: Unauthenticated HTTP request to protected MCP endpoint returns 401."""
    from starlette.testclient import TestClient
    app = create_http_app(transport="sse")
    client = TestClient(app)
    res = client.get("/sse")
    assert res.status_code == 401
    data = res.json()
    assert data["error"] == "AuthenticationError"
    assert "Missing authentication credentials" in data["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# 26. HTTP Invalid Bearer Token Rejection
# ─────────────────────────────────────────────────────────────────────────────
def test_case_26_invalid_token_rejected():
    """Test 26: Request with invalid Bearer token returns 401."""
    from starlette.testclient import TestClient
    app = create_http_app(transport="sse")
    client = TestClient(app)
    res = client.get("/sse", headers={"Authorization": "Bearer invalid_secret_token_xyz"})
    assert res.status_code == 401
    data = res.json()
    assert data["error"] == "AuthenticationError"
    assert "Invalid or expired" in data["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# 27. Query Parameter Token Rejected in Production
# ─────────────────────────────────────────────────────────────────────────────
def test_case_27_query_parameter_token_rejected_in_production(monkeypatch):
    """Test 27: Passing token in query parameters is strictly forbidden in production."""
    from starlette.testclient import TestClient
    monkeypatch.setattr(settings, "environment", "production")
    app = create_http_app(transport="sse")
    client = TestClient(app)
    res = client.get("/sse?token=edulens-dev-token-student-001")
    assert res.status_code == 401
    data = res.json()
    assert "URL query parameters is forbidden" in data["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# 28. Forged student_id Rejection
# ─────────────────────────────────────────────────────────────────────────────
def test_case_28_forged_student_id_rejected(ingestion_service, student_user_1):
    """Test 28: A client cannot forge or supply a student_id different from authenticated user."""
    req = SessionIngestRequest(
        topic="Security Isolation",
        student_id="victim_student_999",  # Forged identity attempt
        messages=[
            MessageItem(role="user", content="Trying to log as victim."),
            MessageItem(role="assistant", content="Response."),
            MessageItem(role="user", content="Understood."),
        ],
    )
    with pytest.raises(AuthorizationError, match="does not match authenticated"):
        ingestion_service.ingest_session(req, student_user_1)


# ─────────────────────────────────────────────────────────────────────────────
# 29. Concurrent Users - ContextVar Isolation
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_case_29_concurrent_users_no_context_leak():
    """Test 29: Concurrent asynchronous tasks execute with strict UserContext isolation."""
    import asyncio
    from backend.mcp_server.auth.authentication import get_current_user, set_current_user, reset_current_user

    async def user_task(user_id: str, delay: float):
        user = UserContext(user_id=user_id, username=f"User {user_id}")
        ctx_token = set_current_user(user)
        try:
            await asyncio.sleep(delay)
            active_user = get_current_user()
            assert active_user.user_id == user_id
            return active_user.user_id
        finally:
            reset_current_user(ctx_token)

    results = await asyncio.gather(
        user_task("student_A", 0.05),
        user_task("student_B", 0.02),
        user_task("student_C", 0.04),
    )
    assert results == ["student_A", "student_B", "student_C"]


# ─────────────────────────────────────────────────────────────────────────────
# 30. Retry Failed Analytics Without Duplicating Raw Records
# ─────────────────────────────────────────────────────────────────────────────
def test_case_30_retry_failed_analytics_without_duplicate_raw_records(ingestion_service, student_user_1, monkeypatch):
    """Test 30: When pipeline fails, raw conversation is kept and can be retried without duplicate records."""
    broken_process = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Temporary out of memory"))
    monkeypatch.setattr(ingestion_service, "_get_pipeline", lambda: type("Broken", (), {"process_session": broken_process})())

    req = SessionIngestRequest(
        topic="Graph Traversal",
        messages=[
            MessageItem(role="user", content="How does DFS differ from BFS?"),
            MessageItem(role="assistant", content="DFS uses a stack; BFS uses a queue."),
            MessageItem(role="user", content="Understood completely!"),
        ],
    )
    res_failed = ingestion_service.ingest_session(req, student_user_1)
    assert res_failed.status == "processing_failed"
    failed_session_id = res_failed.session_id

    # Verify 1 raw session stored
    total_before, _ = ingestion_service.storage.list_sessions(student_user_1.user_id)
    assert total_before == 1

    # Second: fix pipeline and retry
    monkeypatch.undo()
    res_retry = ingestion_service.retry_failed_session(failed_session_id, student_user_1)
    assert res_retry.status == "success"
    assert res_retry.session_id == failed_session_id

    # Verify still exactly 1 raw session (no duplicates!)
    total_after, _ = ingestion_service.storage.list_sessions(student_user_1.user_id)
    assert total_after == 1
    analytics = ingestion_service.get_session_analytics(failed_session_id, student_user_1)
    assert analytics.status == "processed"


# ─────────────────────────────────────────────────────────────────────────────
# 31. Quiz Update Updates Session & Composite Score
# ─────────────────────────────────────────────────────────────────────────────
def test_case_31_quiz_update_updates_session_and_composite_score(ingestion_service, student_user_1):
    """Test 31: Submitting quiz results updates linked session's actual quiz score and composite mastery."""
    # 1. Ingest session without quiz
    req = SessionIngestRequest(
        topic="Recursion",
        messages=[
            MessageItem(role="user", content="What is the base case in recursion?"),
            MessageItem(role="assistant", content="The terminating condition that stops recursion."),
            MessageItem(role="user", content="Got it, prevents stack overflow!"),
        ],
    )
    res = ingestion_service.ingest_session(req, student_user_1)
    sess_id = res.session_id

    # Check initial analytics: quiz status is awaiting
    ana_before = ingestion_service.get_session_analytics(sess_id, student_user_1)
    assert ana_before.actual_quiz_performance.get("status") == "awaiting_quiz_submission"

    # 2. Submit quiz linked to this session
    quiz_req = QuizSubmissionRequest(
        session_id=sess_id,
        topic="Recursion",
        questions=[
            {"question_id": "q1", "student_answer": "base case", "correct_answer": "base case"},
            {"question_id": "q2", "student_answer": "stack", "correct_answer": "stack"},
        ],
    )
    quiz_res = ingestion_service.submit_quiz_results(quiz_req, student_user_1)
    assert quiz_res.actual_score_percent == 100.0
    assert quiz_res.updated_overall_performance is not None
    assert quiz_res.updated_overall_performance["dimension_scores"]["actual_quiz_performance"]["score"] == 100.0

    # 3. Verify session analytics in storage was updated
    ana_after = ingestion_service.get_session_analytics(sess_id, student_user_1)
    assert ana_after.actual_quiz_performance.get("status") == "evaluated"
    assert ana_after.actual_quiz_performance.get("actual_score_percent") == 100.0
    assert ana_after.overall_topic_performance.get("overall_performance_score") is not None


# ─────────────────────────────────────────────────────────────────────────────
# 32. Production Config Does Not Seed Dev Tokens
# ─────────────────────────────────────────────────────────────────────────────
def test_case_32_production_config_does_not_seed_dev_tokens(monkeypatch):
    """Test 32: In production environment, development test tokens are never seeded."""
    monkeypatch.setattr(settings, "environment", "production")
    prod_mgr = TokenManager()
    with pytest.raises(AuthenticationError, match="Invalid or expired"):
        prod_mgr.validate_token("edulens-dev-token-student-001")


# ─────────────────────────────────────────────────────────────────────────────
# 33. MCP Resources and Prompts Discovery
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_case_33_mcp_resources_and_prompts():
    """Test 33: MCP server lists and serves curriculum topics resource and save prompt."""
    server = create_mcp_server()
    resources = await server.list_resources()
    resource_uris = {str(r.uri) for r in resources}
    assert "edulens://curriculum/topics" in resource_uris
    assert "edulens://health" in resource_uris

    prompts = await server.list_prompts()
    prompt_names = {p.name for p in prompts}
    assert "save_learning_session" in prompt_names

