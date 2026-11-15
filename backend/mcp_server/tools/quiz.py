"""
quiz.py — Tool 6: submit_quiz_results for EduLens MCP Server.
"""

from typing import Any, Dict, List, Optional
from backend.mcp_server.auth.authentication import (
    authenticate_token,
    get_current_user,
)
from backend.mcp_server.schemas.session_schema import QuizSubmissionRequest
from backend.mcp_server.services.ingestion_service import get_ingestion_service


def submit_quiz_results(
    questions: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    answer_key: Optional[Dict[str, Any]] = None,
    student_answers: Optional[Dict[str, Any]] = None,
    auth_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Submits quiz questions, answer keys, and student answers for deterministic grading.
    If linked to a session_id, updates the session's overall topic performance score.

    Args:
        questions: List of question objects, each with 'question_id', 'student_answer',
                   and 'correct_answer' (or options and answer keys).
        session_id: Optional learning session ID to link this quiz to.
        subject: Optional academic subject name.
        topic: Optional topic name.
        answer_key: Optional dictionary mapping question IDs to correct answers.
        student_answers: Optional dictionary mapping question IDs to student selected answers.
        auth_token: Optional EduLens API token if not provided in HTTP headers.

    Returns:
        Deterministic grading results:
        - total_questions: Count of evaluated questions
        - correct_count: Number of correct answers
        - incorrect_count: Number of incorrect answers
        - unanswered_count: Number of unanswered questions
        - actual_score_percent: Deterministic score percentage (0-100%)
        - question_breakdown: Item-by-item correct/incorrect evaluation
        - topic_breakdown: Topic-level accuracy breakdown
        - updated_overall_performance: Recalculated composite performance if linked to session
    """
    if auth_token:
        user = authenticate_token(auth_token)
    else:
        user = get_current_user()

    req = QuizSubmissionRequest(
        questions=questions,
        session_id=session_id.strip() if session_id else None,
        subject=subject.strip() if subject else None,
        topic=topic.strip() if topic else None,
        answer_key=answer_key,
        student_answers=student_answers,
    )

    service = get_ingestion_service()
    res = service.submit_quiz_results(req, user)
    return res.model_dump()

