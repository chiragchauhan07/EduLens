"""
server.py — Main EduLens Model Context Protocol (MCP) Server.
Exposes tools, resources, and prompts over Streamable HTTP, SSE, and Stdio transports.
"""

import argparse
import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path

# Ensure ROOT_DIR is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mcp.server.mcpserver import MCPServer
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from backend.mcp_server.config import settings
from backend.mcp_server.auth.authentication import (
    AuthenticationError,
    set_current_user,
    token_manager,
    get_current_user,
)
from backend.mcp_server.tools.ingestion import ingest_learning_session
from backend.mcp_server.tools.analytics import (
    get_session_analytics,
    get_student_performance,
    get_topic_analytics,
)
from backend.mcp_server.tools.sessions import list_learning_sessions
from backend.mcp_server.tools.quiz import submit_quiz_results
from backend.mcp_server.services.storage_service import get_storage_service

logger = logging.getLogger("edulens.mcp")


def create_mcp_server() -> MCPServer:
    """
    Constructs and configures the official EduLens MCPServer instance with all tools,
    resources, and prompts.
    """
    server = MCPServer(
        name=settings.server_name,
        version=settings.server_version,
        instructions=settings.server_description,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 1. CORE MCP TOOLS REGISTRATION
    # ─────────────────────────────────────────────────────────────────────────

    @server.tool(
        name="ingest_learning_session",
        description=(
            "Submits a student and AI-agent conversation to EduLens for learning performance intelligence. "
            "Analyzes topic understanding, extracts conversational signals, estimates confidence, and returns "
            "structured analytics across 4 dimensions."
        ),
    )
    def tool_ingest_learning_session(
        messages: List[Dict[str, Any]],
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        session_id: Optional[str] = None,
        student_id: Optional[str] = None,
        source_agent: str = "chatgpt",
        metadata: Optional[Dict[str, Any]] = None,
        quiz_submission: Optional[Dict[str, Any]] = None,
        auth_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        return ingest_learning_session(
            messages=messages,
            subject=subject,
            topic=topic,
            session_id=session_id,
            student_id=student_id,
            source_agent=source_agent,
            metadata=metadata,
            quiz_submission=quiz_submission,
            auth_token=auth_token,
        )

    @server.tool(
        name="get_session_analytics",
        description=(
            "Retrieves full 4-dimension learning analytics for a previously ingested learning session by session_id."
        ),
    )
    def tool_get_session_analytics(
        session_id: str,
        auth_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        return get_session_analytics(session_id=session_id, auth_token=auth_token)

    @server.tool(
        name="get_student_performance",
        description=(
            "Retrieves accumulated longitudinal performance across all learning sessions for a student: "
            "mastery breakdown, areas of confusion, average scores, and progress timeline."
        ),
    )
    def tool_get_student_performance(
        student_id: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        return get_student_performance(student_id=student_id, auth_token=auth_token)

    @server.tool(
        name="get_topic_analytics",
        description=(
            "Retrieves topic-specific learning analytics across the student's historical sessions for a given topic."
        ),
    )
    def tool_get_topic_analytics(
        topic: str,
        student_id: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        return get_topic_analytics(topic=topic, student_id=student_id, auth_token=auth_token)

    @server.tool(
        name="list_learning_sessions",
        description=(
            "Lists previously ingested learning sessions for the student with optional subject, topic, and date filters."
        ),
    )
    def tool_list_learning_sessions(
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        auth_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        return list_learning_sessions(
            subject=subject,
            topic=topic,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            offset=offset,
            auth_token=auth_token,
        )

    @server.tool(
        name="submit_quiz_results",
        description=(
            "Submits assessment questions, student answers, and answer keys to EduLens for deterministic grading. "
            "Calculates accuracy, breakdown, and updates overall session performance if linked to a session_id."
        ),
    )
    def tool_submit_quiz_results(
        questions: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        answer_key: Optional[Dict[str, Any]] = None,
        student_answers: Optional[Dict[str, Any]] = None,
        auth_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        return submit_quiz_results(
            questions=questions,
            session_id=session_id,
            subject=subject,
            topic=topic,
            answer_key=answer_key,
            student_answers=student_answers,
            auth_token=auth_token,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. RESOURCES REGISTRATION
    # ─────────────────────────────────────────────────────────────────────────

    @server.resource(
        uri="edulens://curriculum/topics",
        name="Curriculum Topics",
        description="List of verified academic topics and subjects analyzed in EduLens",
        mime_type="application/json",
    )
    def resource_curriculum_topics() -> str:
        dataset_path = ROOT_DIR / "data" / "raw" / "edulens_chats_dataset.json"
        if dataset_path.exists():
            with open(dataset_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            topics = sorted(list(set(r.get("topic") for r in records if r.get("topic"))))
            return json.dumps({"topics": topics, "count": len(topics)})
        return json.dumps({"topics": [], "count": 0})

    @server.resource(
        uri="edulens://health",
        name="Health Status",
        description="Server runtime status, version, and loaded dimensions",
        mime_type="application/json",
    )
    def resource_server_health() -> str:
        return json.dumps({
            "status": "healthy",
            "server": settings.server_name,
            "version": settings.server_version,
            "dimensions": [
                "1_topic_understanding",
                "2_actual_quiz_performance",
                "3_expressed_confidence",
                "4_overall_topic_performance",
            ],
            "storage_backend": "sqlite",
            "auth_enabled": settings.auth_enabled,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # 3. PROMPTS REGISTRATION
    # ─────────────────────────────────────────────────────────────────────────

    @server.prompt(
        name="save_learning_session",
        description="Prompt instructing the AI assistant to package and submit the current learning session to EduLens",
    )
    def prompt_save_learning_session(topic: str = "General", subject: str = "Computer Science") -> str:
        return (
            f"You are helping the student record their learning session on '{topic}' ({subject}) in EduLens. "
            "Please format the conversation history into chronological messages with 'role' ('user' for student, "
            "'assistant' for AI) and 'content'. Then invoke the EduLens MCP tool 'ingest_learning_session' with the messages, "
            f"topic='{topic}', and subject='{subject}'. Report the resulting learning analytics to the student."
        )

    return server


import urllib.parse


class AuthMiddleware:
    """
    Raw ASGI middleware verifying Bearer token in HTTP header or query parameters
    and populating UserContext into async task context without buffering response streams.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path in ("/", "/health", "/docs", "/openapi.json"):
                await self.app(scope, receive, send)
                return

            headers = dict(scope.get("headers", []))
            auth_header = headers.get(b"authorization", b"").decode("utf-8")
            query_str = scope.get("query_string", b"").decode("utf-8")
            params = dict(urllib.parse.parse_qsl(query_str))
            token = auth_header or params.get("token") or params.get("api_key")

            try:
                user = token_manager.validate_token(token)
                set_current_user(user)
            except AuthenticationError as auth_err:
                response = JSONResponse(
                    {
                        "error": "AuthenticationError",
                        "detail": str(auth_err),
                        "hint": "Set Authorization: Bearer <token> or append ?token=<token>",
                    },
                    status_code=401,
                )
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)


def create_http_app(transport: str = "sse") -> Starlette:
    """
    Wraps the MCP server inside a Starlette HTTP application with
    authentication, health checks, and CORS headers.
    """
    mcp_server = create_mcp_server()

    if transport.lower() in ("streamable", "streamable_http"):
        app = mcp_server.streamable_http_app()
    else:
        app = mcp_server.sse_app()

    # Add custom diagnostic endpoints
    async def health_endpoint(request: Request):
        return JSONResponse({
            "status": "healthy",
            "server": settings.server_name,
            "version": settings.server_version,
            "transport": transport,
            "auth_enabled": settings.auth_enabled,
        })

    app.routes.insert(0, Route("/health", health_endpoint, methods=["GET"]))
    app.routes.insert(0, Route("/", health_endpoint, methods=["GET"]))

    # Add Middleware
    app.add_middleware(AuthMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


# Global MCP server instance
mcp_server_instance = create_mcp_server()


def main():
    """CLI Entrypoint for running the EduLens MCP Server."""
    parser = argparse.ArgumentParser(description="EduLens Model Context Protocol (MCP) Server")
    parser.add_argument(
        "--transport",
        choices=["sse", "streamable", "stdio"],
        default=settings.transport,
        help="Transport mechanism: 'sse', 'streamable', or 'stdio' (default: sse)",
    )
    parser.add_argument("--host", default=settings.host, help="Bind host address")
    parser.add_argument("--port", type=int, default=settings.port, help="Bind port")
    parser.add_argument("--no-auth", action="store_true", help="Disable token authentication for local testing")

    args = parser.parse_args()

    if args.no_auth:
        settings.auth_enabled = False
        print("[INFO] Authentication disabled via --no-auth flag. Running in open local dev mode.", file=sys.stderr)

    if args.transport == "stdio":
        print("[INFO] Starting EduLens MCP Server on stdio transport (Claude Desktop mode)...", file=sys.stderr)
        server = create_mcp_server()
        asyncio.run(server.run_stdio_async())
    else:
        import uvicorn
        print(f"[INFO] Starting EduLens MCP Server on {args.transport.upper()} transport at http://{args.host}:{args.port}...", file=sys.stderr)
        app = create_http_app(transport=args.transport)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
