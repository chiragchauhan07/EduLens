"""
test_mcp_e2e.py — End-to-End Integration Test for EduLens MCP Integration.
Connects an official MCP ClientSession over Stdio transport to the EduLens server process,
submits a real learning conversation, executes the ML analytics pipeline, and retrieves intelligence.
"""

import asyncio
import json
import sys
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.anyio
async def test_mcp_e2e_stdio_full_workflow():
    """
    End-to-end test verifying full workflow through an official MCP client:
    1. Client initialization and protocol handshake.
    2. Tool discovery verification.
    3. Conversation ingestion via ingest_learning_session tool.
    4. Retrieval of analytics via get_session_analytics tool.
    5. Quiz evaluation via submit_quiz_results tool.
    6. Student performance retrieval via get_student_performance tool.
    7. Session listing via list_learning_sessions tool.
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "backend.mcp_server.server", "--transport", "stdio", "--no-auth"],
        env=None,
    )

    import uuid
    run_id = uuid.uuid4().hex[:8]

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 1. Protocol Handshake
            init_res = await session.initialize()
            assert init_res.server_info.name == "edulens-mcp-server"

            # 2. Tool Discovery
            tools_res = await session.list_tools()
            tool_names = [t.name for t in tools_res.tools]
            assert "ingest_learning_session" in tool_names
            assert "get_session_analytics" in tool_names
            assert "submit_quiz_results" in tool_names
            assert "get_student_performance" in tool_names

            # 3. Ingestion of a real learning dialogue
            sample_messages = [
                {"role": "user", "content": f"How does quicksort partition work? (session {run_id})"},
                {"role": "assistant", "content": "Quicksort selects a pivot and places smaller elements left and larger elements right."},
                {"role": "user", "content": "Oh, so smaller go left and larger go right? Got it, that makes sense now!"},
            ]
            ingest_call = await session.call_tool(
                "ingest_learning_session",
                arguments={
                    "messages": sample_messages,
                    "subject": "Algorithms",
                    "topic": "Quicksort Partitioning",
                    "source_agent": "chatgpt",
                },
            )
            assert ingest_call.content is not None
            assert len(ingest_call.content) > 0

            ingest_data = json.loads(ingest_call.content[0].text)
            assert ingest_data["status"] == "success"
            session_id = ingest_data["session_id"]
            assert session_id.startswith("SES_")
            assert ingest_data["topic"] == "Quicksort Partitioning"

            # 4. Analytics Retrieval
            ana_call = await session.call_tool(
                "get_session_analytics",
                arguments={"session_id": session_id},
            )
            ana_data = json.loads(ana_call.content[0].text)
            assert ana_data["session_id"] == session_id
            assert ana_data["topic_understanding"]["predicted_class"] in ("understood", "partial", "confused")
            assert ana_data["learning_confidence"]["expressed_confidence_level"] in ("low", "moderate", "high")

            # 5. Quiz Submission
            quiz_call = await session.call_tool(
                "submit_quiz_results",
                arguments={
                    "session_id": session_id,
                    "topic": "Quicksort Partitioning",
                    "questions": [
                        {"question_id": "q1", "student_answer": "pivot", "correct_answer": "pivot"},
                        {"question_id": "q2", "student_answer": "O(n log n)", "correct_answer": "O(n log n)"},
                    ],
                },
            )
            quiz_data = json.loads(quiz_call.content[0].text)
            assert quiz_data["actual_score_percent"] == 100.0
            assert quiz_data["correct_count"] == 2

            # 6. Student Performance Portfolio
            perf_call = await session.call_tool(
                "get_student_performance",
                arguments={},
            )
            perf_data = json.loads(perf_call.content[0].text)
            assert perf_data["total_sessions"] >= 1
            assert "Quicksort Partitioning" in perf_data["topics_studied"]

            # 7. List Learning Sessions
            list_call = await session.call_tool(
                "list_learning_sessions",
                arguments={"limit": 10},
            )
            list_data = json.loads(list_call.content[0].text)
            assert list_data["total_count"] >= 1
            assert any(s["session_id"] == session_id for s in list_data["sessions"])
