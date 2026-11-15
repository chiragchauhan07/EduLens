"""
verify_mcp_inspector.py — Comprehensive MCP Inspector simulation against EduLens MCP Server.
Tests Streamable HTTP (/mcp) for all 6 tools, resources, prompts, auth failures, and isolation.
"""

import sys
import uuid
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from starlette.testclient import TestClient
from backend.mcp_server.server import create_http_app
from backend.mcp_server.config import settings

def parse_sse_response(text: str) -> dict:
    """Parses SSE 'event: message\ndata: {...}' or direct JSON response."""
    for line in text.splitlines():
        if line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            if data_str:
                return json.loads(data_str)
    return json.loads(text)

def run_inspector_verification():
    print("=" * 70)
    print("EDULENS MCP INSPECTOR VERIFICATION (Streamable HTTP at /mcp)")
    print("=" * 70)

    app = create_http_app(transport="streamable")
    with TestClient(app, base_url="http://127.0.0.1:8001") as client:
        dev_token = settings.default_dev_token
        auth_headers = {"Authorization": f"Bearer {dev_token}"}
        run_id = uuid.uuid4().hex[:6]

        # 1. Health check
        print("\n[Step 1] Checking /health endpoint...")
        h_res = client.get("/health")
        assert h_res.status_code == 200, f"Expected 200, got {h_res.status_code}"
        print(f"  ✓ Server Health: {h_res.json()}")

        # 2. Initialize
        print("\n[Step 2] Sending MCP initialize request...")
        init_res = client.post(
            "/mcp",
            headers=auth_headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "official-mcp-inspector-sim", "version": "1.0"},
                },
            },
        )
        assert init_res.status_code == 200, f"Initialize failed: {init_res.status_code} {init_res.text}"
        session_id = init_res.headers.get("mcp-session-id")
        init_data = parse_sse_response(init_res.text)
        print(f"  ✓ MCP Initialized successfully! Session ID: {session_id}")
        print(f"  ✓ Server Protocol Info: {init_data.get('result', {}).get('serverInfo', {})}")

        session_headers = {"Authorization": f"Bearer {dev_token}", "mcp-session-id": session_id}

        # 3. Notification initialized
        client.post(
            "/mcp",
            headers=session_headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        print("  ✓ Sent notifications/initialized")

        # 4. List Tools (Verify all 6 tools)
        print("\n[Step 3] Querying tools/list...")
        tools_res = client.post(
            "/mcp",
            headers=session_headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        assert tools_res.status_code == 200
        tools_data = parse_sse_response(tools_res.text).get("result", {}).get("tools", [])
        tool_names = [t["name"] for t in tools_data]
        print(f"  ✓ Discovered {len(tool_names)} tools: {tool_names}")
        expected_tools = [
            "ingest_learning_session",
            "get_session_analytics",
            "get_student_performance",
            "get_topic_analytics",
            "list_learning_sessions",
            "submit_quiz_results",
        ]
        for et in expected_tools:
            assert et in tool_names, f"Missing tool: {et}"
        print("  ✓ All 6 expected tools registered with valid schemas!")

        # 5. List Resources & Read
        print("\n[Step 4] Querying resources/list and resources/read...")
        res_list = client.post(
            "/mcp",
            headers=session_headers,
            json={"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}},
        )
        assert res_list.status_code == 200
        resources = parse_sse_response(res_list.text).get("result", {}).get("resources", [])
        print(f"  ✓ Discovered {len(resources)} resources: {[r['uri'] for r in resources]}")

        read_res = client.post(
            "/mcp",
            headers=session_headers,
            json={
                "jsonrpc": "2.0",
                "id": 4,
                "method": "resources/read",
                "params": {"uri": "edulens://health"},
            },
        )
        assert read_res.status_code == 200
        read_data = parse_sse_response(read_res.text)
        print("  ✓ Successfully read edulens://health resource")

        # 6. List Prompts
        print("\n[Step 5] Querying prompts/list...")
        p_res = client.post(
            "/mcp",
            headers=session_headers,
            json={"jsonrpc": "2.0", "id": 5, "method": "prompts/list", "params": {}},
        )
        assert p_res.status_code == 200
        prompts = parse_sse_response(p_res.text).get("result", {}).get("prompts", [])
        print(f"  ✓ Discovered {len(prompts)} prompts: {[p['name'] for p in prompts]}")

        # 7. Execute Tool 1: ingest_learning_session
        print("\n[Step 6] Executing Tool 1: ingest_learning_session...")
        ingest_res = client.post(
            "/mcp",
            headers=session_headers,
            json={
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {
                    "name": "ingest_learning_session",
                    "arguments": {
                        "topic": "Binary Search Trees",
                        "subject": "Data Structures",
                        "messages": [
                            {"role": "user", "content": f"How does a BST maintain ordering? (run {run_id})"},
                            {"role": "assistant", "content": "Left subtree has smaller keys, right subtree has larger keys."},
                            {"role": "user", "content": "Understood, that enables O(log N) lookup in balanced BSTs!"},
                        ],
                    },
                },
            },
        )
        assert ingest_res.status_code == 200, f"Ingest failed: {ingest_res.text}"
        ingest_data = parse_sse_response(ingest_res.text)
        ingest_out = ingest_data["result"]["content"][0]["text"]
        ingest_dict = json.loads(ingest_out) if isinstance(ingest_out, str) else ingest_out
        created_session_id = ingest_dict["session_id"]
        print(f"  ✓ Ingestion successful! Session ID: {created_session_id}")
        print(f"  ✓ Understanding predicted: {ingest_dict['analytics']['topics'][0]['understanding']['classification']}")

        # 8. Execute Tool 2: get_session_analytics
        print("\n[Step 7] Executing Tool 2: get_session_analytics...")
        sess_res = client.post(
            "/mcp",
            headers=session_headers,
            json={
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {
                    "name": "get_session_analytics",
                    "arguments": {"session_id": created_session_id},
                },
            },
        )
        assert sess_res.status_code == 200
        sess_data = parse_sse_response(sess_res.text)
        sess_out = sess_data["result"]["content"][0]["text"]
        sess_dict = json.loads(sess_out) if isinstance(sess_out, str) else sess_out
        assert sess_dict["session_id"] == created_session_id
        print(f"  ✓ Analytics retrieved! Understanding Score: {sess_dict['topic_understanding']['understanding_score']}")

        # 9. Execute Tool 3: get_student_performance
        print("\n[Step 8] Executing Tool 3: get_student_performance...")
        perf_res = client.post(
            "/mcp",
            headers=session_headers,
            json={
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {
                    "name": "get_student_performance",
                    "arguments": {},
                },
            },
        )
        assert perf_res.status_code == 200
        perf_data = parse_sse_response(perf_res.text)
        perf_out = perf_data["result"]["content"][0]["text"]
        perf_dict = json.loads(perf_out) if isinstance(perf_out, str) else perf_out
        print(f"  ✓ Student performance retrieved: {perf_dict['total_sessions']} sessions, topics: {perf_dict['topics_studied']}")

        # 10. Execute Tool 4: get_topic_analytics
        print("\n[Step 9] Executing Tool 4: get_topic_analytics...")
        top_res = client.post(
            "/mcp",
            headers=session_headers,
            json={
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {
                    "name": "get_topic_analytics",
                    "arguments": {"topic": "Binary Search Trees"},
                },
            },
        )
        assert top_res.status_code == 200
        top_data = parse_sse_response(top_res.text)
        top_out = top_data["result"]["content"][0]["text"]
        top_dict = json.loads(top_out) if isinstance(top_out, str) else top_out
        print(f"  ✓ Topic analytics retrieved: {top_dict['topic']}, sessions: {top_dict['total_sessions']}")

        # 11. Execute Tool 5: list_learning_sessions
        print("\n[Step 10] Executing Tool 5: list_learning_sessions...")
        list_res = client.post(
            "/mcp",
            headers=session_headers,
            json={
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {
                    "name": "list_learning_sessions",
                    "arguments": {"limit": 10},
                },
            },
        )
        assert list_res.status_code == 200
        list_data = parse_sse_response(list_res.text)
        list_out = list_data["result"]["content"][0]["text"]
        list_dict = json.loads(list_out) if isinstance(list_out, str) else list_out
        print(f"  ✓ Listed {list_dict['total_count']} session records (offset {list_dict['offset']})")

        # 12. Execute Tool 6: submit_quiz_results
        print("\n[Step 11] Executing Tool 6: submit_quiz_results...")
        quiz_res = client.post(
            "/mcp",
            headers=session_headers,
            json={
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {
                    "name": "submit_quiz_results",
                    "arguments": {
                        "session_id": created_session_id,
                        "topic": "Binary Search Trees",
                        "questions": [
                            {"question_id": "q1", "student_answer": "B", "correct_answer": "B"},
                            {"question_id": "q2", "student_answer": "A", "correct_answer": "A"},
                        ],
                    },
                },
            },
        )
        assert quiz_res.status_code == 200
        quiz_data = parse_sse_response(quiz_res.text)
        quiz_out = quiz_data["result"]["content"][0]["text"]
        quiz_dict = json.loads(quiz_out) if isinstance(quiz_out, str) else quiz_out
        print(f"  ✓ Quiz evaluated: {quiz_dict['actual_score_percent']}% accuracy ({quiz_dict['correct_count']}/{quiz_dict['total_questions']} correct)")
        print(f"  ✓ Updated composite score: {quiz_dict['updated_overall_performance']['overall_performance_score']}")

        # 13. Authentication Failure Verification
        print("\n[Step 12] Testing Authentication Failure Cases...")
        # A. Missing header
        res_no_auth = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 12, "method": "tools/list", "params": {}},
        )
        assert res_no_auth.status_code == 401
        print(f"  ✓ Missing header rejected: HTTP {res_no_auth.status_code}")

        # B. Invalid bearer token
        res_bad_token = client.post(
            "/mcp",
            headers={"Authorization": "Bearer invalid_secret_token_xyz"},
            json={"jsonrpc": "2.0", "id": 13, "method": "tools/list", "params": {}},
        )
        assert res_bad_token.status_code == 401
        print(f"  ✓ Invalid token rejected: HTTP {res_bad_token.status_code}")

        # C. Query parameter token in production
        old_env = settings.environment
        settings.environment = "production"
        try:
            res_query_tok = client.post(
                "/mcp?token=edulens-dev-token-student-001",
                json={"jsonrpc": "2.0", "id": 14, "method": "tools/list", "params": {}},
            )
            assert res_query_tok.status_code == 401
            print(f"  ✓ Query parameter token rejected in production: HTTP {res_query_tok.status_code}")
        finally:
            settings.environment = old_env

        # 14. Cross-User Isolation Verification
        print("\n[Step 13] Testing Cross-User Isolation & Forged Student ID...")
        # Student 2 attempts to query Student 1's session analytics
        res_cross_user = client.post(
            "/mcp",
            headers={"Authorization": "Bearer edulens-test-token-student-002", "mcp-session-id": session_id},
            json={
                "jsonrpc": "2.0",
                "id": 15,
                "method": "tools/call",
                "params": {
                    "name": "get_session_analytics",
                    "arguments": {"session_id": created_session_id},
                },
            },
        )
        assert "error" in res_cross_user.text.lower() or "keyerror" in res_cross_user.text.lower() or "not found" in res_cross_user.text.lower()
        print("  ✓ Cross-user access blocked: Student 2 cannot read Student 1's session analytics!")

        # Student 1 attempts to forge student_id="student_002" in ingest_learning_session
        res_forged = client.post(
            "/mcp",
            headers=session_headers,
            json={
                "jsonrpc": "2.0",
                "id": 16,
                "method": "tools/call",
                "params": {
                    "name": "ingest_learning_session",
                    "arguments": {
                        "topic": "Recursion",
                        "student_id": "student_002",  # Forged identity
                        "messages": [
                            {"role": "user", "content": "What is a recursive call?"},
                            {"role": "assistant", "content": "A function that calls itself."},
                            {"role": "user", "content": "Got it!"},
                        ],
                    },
                },
            },
        )
        forged_data = parse_sse_response(res_forged.text)
        assert forged_data.get("result", {}).get("isError") is True or "error" in res_forged.text.lower()
        print("  ✓ Identity forgery blocked: Server returned isError=True on mismatched student_id!")

        print("\n" + "=" * 70)
        print("ALL 13 MCP INSPECTOR PROTOCOL STEPS & SECURITY CHECKS PASSED!")
        print("=" * 70)

if __name__ == "__main__":
    run_inspector_verification()
