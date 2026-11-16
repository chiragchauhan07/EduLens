"""
analytics.py — Tools 2, 3, and 4: Analytics retrieval tools for EduLens MCP Server.
"""

from typing import Any, Dict, Optional
from backend.mcp_server.auth.authentication import (
    authenticate_token,
    get_current_user,
)
from backend.mcp_server.services.ingestion_service import get_ingestion_service


def get_session_analytics(
    session_id: str,
    auth_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieves full 4-dimension learning analytics for a previously ingested learning session.

    Args:
        session_id: The unique ID of the session returned during ingestion.
        auth_token: Optional EduLens API token if not provided in HTTP headers.

    Returns:
        Structured analytics report containing:
        - Topic Understanding (predicted class, understanding score, probabilities)
        - Actual Quiz Performance (score percent, correct/incorrect counts)
        - Expressed Learning Confidence (linguistic confidence level and score)
        - Overall Topic Performance (composite score, performance category)
        - Discrepancy Analysis (divergence detection)
        - Pedagogical Recommendation and Feature Explanation
    """
    if auth_token:
        user = authenticate_token(auth_token)
    else:
        user = get_current_user()

    service = get_ingestion_service()
    res = service.get_session_analytics(session_id.strip(), user)
    return res.model_dump()


def get_student_performance(
    student_id: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieves the student's accumulated longitudinal learning performance across all sessions.

    Args:
        student_id: Optional student ID. Defaults to the authenticated student.
        auth_token: Optional EduLens API token if not provided in HTTP headers.

    Returns:
        Comprehensive student portfolio:
        - Total sessions completed
        - Topics studied
        - Understanding distribution (counts of understood, partial, confused)
        - Average understanding score (0-100)
        - Average actual quiz score (if quizzes taken)
        - Average expressed confidence score
        - Average overall performance score
        - Topic-wise breakdown with mastery rates
        - Areas of confusion (topics where the student struggled)
        - Chronological learning progress timeline
    """
    if auth_token:
        user = authenticate_token(auth_token)
    else:
        user = get_current_user()

    service = get_ingestion_service()
    res = service.get_student_performance(student_id.strip() if student_id else None, user)
    return res.model_dump()


def get_topic_analytics(
    topic: str,
    student_id: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieves topic-specific learning analytics across the student's learning history.

    Args:
        topic: The academic topic name to inspect (e.g. 'Binary Search', 'Gradient Descent').
        student_id: Optional student ID. Defaults to the authenticated student.
        auth_token: Optional EduLens API token if not provided in HTTP headers.

    Returns:
        Topic intelligence:
        - Topic name and subject
        - Total sessions on this topic
        - Understanding distribution (understood / partial / confused)
        - Average understanding score
        - Quiz performance history
        - Expressed confidence history
        - Overall topic performance
        - Chronological trend across sessions
    """
    if auth_token:
        user = authenticate_token(auth_token)
    else:
        user = get_current_user()

    service = get_ingestion_service()
    res = service.get_topic_analytics(topic.strip(), student_id.strip() if student_id else None, user)
    return res.model_dump()

