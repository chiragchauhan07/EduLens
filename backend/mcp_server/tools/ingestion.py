"""
ingestion.py — Tool 1: ingest_learning_session for EduLens MCP Server.
"""

from typing import Any, Dict, List, Optional
from backend.mcp_server.auth.authentication import (
    authenticate_token,
    get_current_user,
    AuthenticationError,
    AuthorizationError,
)
from backend.mcp_server.schemas.session_schema import MessageItem, SessionIngestRequest
from backend.mcp_server.services.ingestion_service import get_ingestion_service


def ingest_learning_session(
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
    """
    Submits a student-AI learning conversation to EduLens for learning performance intelligence.

    Call this tool when the student finishes learning a concept or asks to save, analyze, or
    send their study session to EduLens.

    Args:
        messages: Chronological list of message objects, each containing 'role' ('user'/'student'
                  or 'assistant'/'teacher') and 'content' (the message text). Optional 'timestamp'.
        subject: Academic subject name (e.g. 'Computer Science', 'Machine Learning', 'Data Structures').
        topic: Specific concept or topic studied (e.g. 'Logistic Regression', 'Binary Search Trees').
        session_id: Optional custom session ID. Auto-generated if omitted.
        student_id: Optional student ID. Defaults to the authenticated user.
        source_agent: Name of the AI agent platform ('chatgpt', 'claude', 'generic').
        metadata: Optional dictionary with session metadata (e.g. conversation_language, duration).
        quiz_submission: Optional quiz questions, answers, and keys to evaluate deterministically.
        auth_token: Optional EduLens API token if not provided in HTTP headers.

    Returns:
        Structured JSON dictionary containing ingestion status, session_id, and 4-dimension analytics:
        - Topic Understanding (predicted level: understood/partial/confused, 0-100 score, confidence)
        - Actual Quiz Performance (score percent, questions correct/incorrect)
        - Expressed Learning Confidence (linguistic level: low/moderate/high, score)
        - Overall Topic Performance (composite 40/40/20 score, mastery category)
        - Discrepancy Analysis (divergence between conversation and quiz)
        - Formative Pedagogical Feedback
    """
    # 1. Authenticate user
    if auth_token:
        user = authenticate_token(auth_token)
    else:
        user = get_current_user()

    # Verify student_id cannot override authenticated identity
    if student_id and student_id.strip() != user.user_id:
        raise AuthorizationError(
            f"Client supplied student_id '{student_id}' does not match authenticated identity '{user.user_id}'."
        )

    # 2. Convert and validate message items
    parsed_messages: List[MessageItem] = []
    for idx, m in enumerate(messages):
        if not isinstance(m, dict):
            raise ValueError(f"Message at index {idx} must be a dictionary with 'role' and 'content'.")
        if "role" not in m or "content" not in m:
            raise ValueError(f"Message at index {idx} is missing required 'role' or 'content' field.")
        parsed_messages.append(
            MessageItem(
                role=str(m["role"]),
                content=str(m["content"]),
                timestamp=m.get("timestamp"),
            )
        )

    # 3. Build request
    req = SessionIngestRequest(
        messages=parsed_messages,
        subject=subject,
        topic=topic,
        session_id=session_id,
        student_id=student_id or user.user_id,
        source_agent=source_agent,
        metadata=metadata,
        quiz_submission=quiz_submission,
    )

    # 4. Process via Ingestion Service
    service = get_ingestion_service()
    result = service.ingest_session(req, user)
    return result.model_dump()

