"""
session_schema.py — Pydantic validation schemas for EduLens MCP protocol.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class MessageItem(BaseModel):
    """Single conversational turn in a student-AI learning dialogue."""

    role: str = Field(
        ...,
        description="Speaker role: 'user' / 'student' for the learner, or 'assistant' / 'teacher' for the AI agent",
    )
    content: str = Field(
        ...,
        description="Text content of the utterance",
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO-8601 formatted timestamp when the utterance was spoken",
    )

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        clean = v.strip().lower()
        allowed = {"user", "student", "assistant", "teacher", "system", "human", "ai"}
        if clean not in allowed:
            raise ValueError(f"Invalid message role '{v}'. Allowed roles: {sorted(allowed)}")
        return clean

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty.")
        return v.strip()


class SessionIngestRequest(BaseModel):
    """Payload submitted to ingest_learning_session tool."""

    messages: List[MessageItem] = Field(
        ...,
        description="Chronological sequence of student and AI assistant dialogue messages",
    )
    subject: Optional[str] = Field(
        default=None,
        description="Academic subject name (e.g. 'Computer Science', 'Data Structures')",
    )
    topic: Optional[str] = Field(
        default=None,
        description="Specific concept or topic studied (e.g. 'Binary Search Trees', 'Logistic Regression')",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the session (auto-generated if omitted)",
    )
    student_id: Optional[str] = Field(
        default=None,
        description="Student identifier (defaults to authenticated user if omitted)",
    )
    source_agent: str = Field(
        default="chatgpt",
        description="Source AI platform or agent ('chatgpt', 'claude', 'generic', etc.)",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional session metadata (e.g. duration, language, client version)",
    )
    quiz_submission: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional post-session quiz answers and questions to evaluate deterministically",
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: List[MessageItem]) -> List[MessageItem]:
        if not v:
            raise ValueError("Conversation must contain at least one message.")
        # Ensure at least one student utterance is present
        has_student = any(m.role in ("user", "student", "human") for m in v)
        if not has_student:
            raise ValueError("Conversation must contain at least one student utterance ('user' or 'student' role).")
        return v


class SessionIngestResult(BaseModel):
    """Structured response returned by ingest_learning_session tool."""

    status: str = Field(..., description="'success', 'duplicate_detected', or 'error'")
    session_id: str
    user_id: str
    source_agent: str
    subject: Optional[str] = None
    topic: Optional[str] = None
    ingested_at: str
    turns_count: int
    is_duplicate: bool = False
    message: Optional[str] = None
    analytics: Optional[Dict[str, Any]] = None


class SessionAnalyticsResponse(BaseModel):
    """Full analytical intelligence report for an individual learning session."""

    session_id: str
    user_id: str
    subject: Optional[str] = None
    topic: Optional[str] = None
    source_agent: str
    status: str
    topic_understanding: Dict[str, Any]
    actual_quiz_performance: Dict[str, Any]
    learning_confidence: Dict[str, Any]
    overall_topic_performance: Dict[str, Any]
    discrepancy_analysis: Dict[str, Any]
    conversational_signals: Dict[str, Any]
    feature_explanation: Optional[Dict[str, Any]] = None
    pedagogical_feedback: str
    processed_at: str


class StudentPerformanceResponse(BaseModel):
    """Accumulated longitudinal learning performance across student history."""

    student_id: str
    total_sessions: int
    topics_studied: List[str]
    understanding_distribution: Dict[str, int]
    average_understanding_score: float
    average_actual_quiz_score: Optional[float] = None
    average_confidence_score: float
    average_overall_score: Optional[float] = None
    topic_wise_breakdown: List[Dict[str, Any]]
    areas_of_confusion: List[str]
    learning_progress: List[Dict[str, Any]]


class TopicAnalyticsResponse(BaseModel):
    """Aggregate analytics for a specific curriculum topic."""

    topic: str
    subject: Optional[str] = None
    student_id: str
    total_sessions: int
    understanding_distribution: Dict[str, int]
    average_understanding_score: float
    quiz_performance: Dict[str, Any]
    expressed_confidence: Dict[str, Any]
    overall_topic_performance: Optional[Dict[str, Any]] = None
    historical_trend: List[Dict[str, Any]]


class SessionListItem(BaseModel):
    """Summary item for session listing."""

    session_id: str
    user_id: str
    subject: Optional[str] = None
    topic: Optional[str] = None
    source_agent: str
    status: str
    understanding_class: Optional[str] = None
    overall_score: Optional[float] = None
    ingested_at: str


class SessionListResponse(BaseModel):
    """Paginated list of student learning sessions."""

    total_count: int
    limit: int
    offset: int
    sessions: List[SessionListItem]


class QuizSubmissionRequest(BaseModel):
    """Payload for submitting assessment questions and answers."""

    questions: List[Dict[str, Any]] = Field(
        ...,
        description="List of questions with student_answer and correct_answer keys, or options",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Associated learning session ID to link and update overall topic performance",
    )
    subject: Optional[str] = None
    topic: Optional[str] = None
    answer_key: Optional[Dict[str, Any]] = None
    student_answers: Optional[Dict[str, Any]] = None


class QuizSubmissionResult(BaseModel):
    """Result of deterministic quiz grading."""

    quiz_id: str
    session_id: Optional[str] = None
    user_id: str
    total_questions: int
    correct_count: int
    incorrect_count: int
    unanswered_count: int
    actual_score_percent: float
    question_breakdown: List[Dict[str, Any]]
    topic_breakdown: Dict[str, Any]
    evaluated_at: str
    updated_overall_performance: Optional[Dict[str, Any]] = None

