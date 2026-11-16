"""
sessions.py — Tool 5: list_learning_sessions for EduLens MCP Server.
"""

from typing import Any, Dict, Optional
from backend.mcp_server.auth.authentication import (
    authenticate_token,
    get_current_user,
)
from backend.mcp_server.services.ingestion_service import get_ingestion_service


def list_learning_sessions(
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    auth_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Lists previously ingested learning sessions for the authenticated student.

    Args:
        subject: Optional filter by academic subject.
        topic: Optional filter by topic name.
        date_from: Optional ISO-8601 start date filter.
        date_to: Optional ISO-8601 end date filter.
        limit: Maximum number of sessions to return (default 20, max 100).
        offset: Pagination offset (default 0).
        auth_token: Optional EduLens API token if not provided in HTTP headers.

    Returns:
        Paginated session summaries:
        - total_count: Total matching sessions
        - limit: Active limit
        - offset: Active offset
        - sessions: List of session objects with session_id, subject, topic, status,
                    understanding outcome, and overall score
    """
    if auth_token:
        user = authenticate_token(auth_token)
    else:
        user = get_current_user()

    safe_limit = max(1, min(limit, 100))
    safe_offset = max(0, offset)

    service = get_ingestion_service()
    res = service.list_learning_sessions(
        user=user,
        subject=subject.strip() if subject else None,
        topic=topic.strip() if topic else None,
        date_from=date_from.strip() if date_from else None,
        date_to=date_to.strip() if date_to else None,
        limit=safe_limit,
        offset=safe_offset,
    )
    return res.model_dump()

