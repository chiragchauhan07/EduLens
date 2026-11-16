"""
ingestion_service.py — Core ingestion and analytics orchestration service for EduLens MCP.
Glues MCP protocol inputs directly to the existing unified ML, DSP, and NLP pipelines.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ml_prototype.unified_pipeline import EduLensUnifiedPipeline
from ml_prototype.quiz_evaluator import evaluate_quiz_submission
from ml_prototype.scoring_engine import TopicPerformanceScoringEngine

from backend.mcp_server.auth.authentication import AuthorizationError, UserContext
from backend.mcp_server.schemas.session_schema import (
    MessageItem,
    QuizSubmissionRequest,
    QuizSubmissionResult,
    SessionAnalyticsResponse,
    SessionIngestRequest,
    SessionIngestResult,
    SessionListItem,
    SessionListResponse,
    StudentPerformanceResponse,
    TopicAnalyticsResponse,
)
from backend.mcp_server.services.storage_service import StorageService, get_storage_service


class IngestionService:
    """
    Orchestrates conversation ingestion, raw JSON storage, idempotency verification,
    and invocation of the unified learning analytics pipeline.
    """

    def __init__(self, storage: Optional[StorageService] = None):
        self.storage = storage or get_storage_service()
        self._pipeline: Optional[EduLensUnifiedPipeline] = None

    def _get_pipeline(self) -> EduLensUnifiedPipeline:
        """Lazy loads and caches the unified ML pipeline."""
        if self._pipeline is None:
            self._pipeline = EduLensUnifiedPipeline(algorithm="logistic_regression")
        return self._pipeline

    @staticmethod
    def _compute_content_hash(messages: List[Any], user_id: str) -> str:
        """Computes a deterministic SHA-256 hash of student messages for idempotency."""
        normalized_content = []
        for m in messages:
            role = m.role if hasattr(m, "role") else m.get("role", "")
            content = m.content if hasattr(m, "content") else m.get("content", "")
            normalized_content.append(f"{role.strip().lower()}:{content.strip()}")
        raw_key = f"{user_id}::" + "||".join(normalized_content)
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def ingest_session(self, request: SessionIngestRequest, user: UserContext) -> SessionIngestResult:
        """
        Ingests a learning conversation:
        1. Validates payload and role integrity.
        2. Computes content hash and detects duplicate submissions.
        3. Persists original raw JSON session to separate raw storage.
        4. Invokes the unified analytics pipeline directly.
        5. Persists analytics results and returns structured intelligence.
        """
        # User isolation: use authenticated user identity
        effective_user_id = user.user_id

        # Verify client-supplied student_id cannot forge or override authenticated identity
        if request.student_id and request.student_id.strip() != effective_user_id:
            raise AuthorizationError(
                f"Client supplied student_id '{request.student_id}' does not match authenticated user identity '{effective_user_id}'."
            )

        # Verify idempotency
        content_hash = self._compute_content_hash(request.messages, effective_user_id)
        existing = self.storage.find_session_by_content_hash(content_hash, effective_user_id)
        if existing and existing.get("status") == "processed":
            existing_analytics = self.storage.get_analytics(existing["session_id"], effective_user_id)
            return SessionIngestResult(
                status="duplicate_detected",
                session_id=existing["session_id"],
                user_id=effective_user_id,
                source_agent=existing.get("source_agent", request.source_agent),
                subject=existing.get("subject", request.subject),
                topic=existing.get("topic", request.topic),
                ingested_at=existing.get("ingested_at", datetime.now(timezone.utc).isoformat()),
                turns_count=existing.get("turns_count", len(request.messages)),
                is_duplicate=True,
                message="Identical learning conversation was previously ingested. Returning existing analytics.",
                analytics=existing_analytics.get("full_analytics_json") if existing_analytics else None,
            )

        # Reuse session identifier on retry of previously failed session to avoid duplicate raw records
        if existing and existing.get("status") != "processed":
            session_id = request.session_id or existing["session_id"]
        else:
            session_id = request.session_id or f"SES_{uuid.uuid4().hex[:12].upper()}"

        # Filter out system messages / developer instructions for privacy and safety
        cleaned_turns: List[Dict[str, Any]] = []
        student_utterances: List[str] = []

        turn_idx = 1
        for msg in request.messages:
            role = msg.role.strip().lower()
            if role == "system":
                # Exclude system prompts from learning conversation analysis
                continue

            content = msg.content.strip()
            cleaned_turns.append({
                "turn": turn_idx,
                "role": role,
                "content": content,
                "timestamp": msg.timestamp,
            })
            if role in ("user", "student", "human"):
                student_utterances.append(content)
            turn_idx += 1

        if not student_utterances:
            raise ValueError("Conversation must contain at least one valid student utterance ('user' or 'student' role).")

        # Determine effective metadata without inventing false subjects or topics
        effective_subject = request.subject.strip() if request.subject and request.subject.strip() else "Unspecified Subject"
        effective_topic = request.topic.strip() if request.topic and request.topic.strip() else "Unspecified Topic"
        effective_agent = request.source_agent.strip() if request.source_agent and request.source_agent.strip() else "Unspecified Agent"

        # Build raw storage payload
        raw_storage_payload = {
            "session_id": session_id,
            "user_id": effective_user_id,
            "source_agent": effective_agent,
            "subject": effective_subject,
            "topic": effective_topic,
            "content_hash": content_hash,
            "turns_count": len(cleaned_turns),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "status": "ingested",
            "raw_json": {
                "session_id": session_id,
                "user_id": effective_user_id,
                "source_agent": effective_agent,
                "subject": effective_subject,
                "topic": effective_topic,
                "messages": [m.model_dump() for m in request.messages],
                "metadata": request.metadata or {},
                "quiz_submission": request.quiz_submission,
            },
        }

        # 1. Store Raw Session First (Ensures conversation is never lost)
        self.storage.save_raw_session(raw_storage_payload)

        # 2. Prepare pipeline input format compatible with DSP & NLP
        # We supply turns format and student_messages list
        pipeline_input = {
            "chat_id": session_id,
            "student_id": effective_user_id,
            "subject": effective_subject,
            "topic": effective_topic,
            "turns": [
                {
                    "turn": i + 1,
                    "student_message": m["content"] if m["role"] in ("user", "student", "human") else "",
                    "teacher_message": m["content"] if m["role"] in ("assistant", "teacher", "ai") else "",
                }
                for i, m in enumerate(cleaned_turns)
            ],
            "student_messages": student_utterances,
        }

        # 3. Execute Unified Analytics Pipeline
        try:
            pipeline = self._get_pipeline()
            analytics_result = pipeline.process_session(
                session_input=pipeline_input,
                quiz_submission=request.quiz_submission,
                explain=True,
                topic_override=request.topic if request.topic and request.topic.strip() else effective_topic,
            )

            # 4. Save Analytics to Analytics Storage
            self.storage.save_analytics(session_id, effective_user_id, analytics_result)

            return SessionIngestResult(
                status="success",
                session_id=session_id,
                user_id=effective_user_id,
                source_agent=request.source_agent,
                subject=request.subject,
                topic=request.topic,
                ingested_at=raw_storage_payload["ingested_at"],
                turns_count=len(cleaned_turns),
                is_duplicate=False,
                message="Learning conversation successfully ingested and evaluated across 4 dimensions.",
                analytics=analytics_result,
            )

        except Exception as err:
            # Mark raw session as processing_failed, but do not delete raw conversation!
            self.storage.update_session_status(session_id, effective_user_id, "processing_failed", str(err))
            return SessionIngestResult(
                status="processing_failed",
                session_id=session_id,
                user_id=effective_user_id,
                source_agent=request.source_agent,
                subject=request.subject,
                topic=request.topic,
                ingested_at=raw_storage_payload["ingested_at"],
                turns_count=len(cleaned_turns),
                is_duplicate=False,
                message=f"Conversation stored securely, but analytics pipeline failed: {str(err)}",
                analytics=None,
            )

    def get_session_analytics(self, session_id: str, user: UserContext) -> SessionAnalyticsResponse:
        """Retrieves previously generated analytics for a session with user-level access verification."""
        analytics = self.storage.get_analytics(session_id, user.user_id)
        if not analytics:
            # Check if session exists in raw_sessions
            raw = self.storage.get_raw_session(session_id, user.user_id)
            if not raw:
                raise KeyError(f"Session '{session_id}' not found for user '{user.user_id}'.")
            raise ValueError(f"Session '{session_id}' exists with status '{raw.get('status')}', but has no completed analytics.")

        raw_json = analytics["full_analytics_json"]
        raw_sess = self.storage.get_raw_session(session_id, user.user_id) or {}

        if "topics" in raw_json and raw_json["topics"]:
            top_topic = raw_json["topics"][0]
            u = top_topic.get("understanding", {}) or {}
            topic_understanding = {
                "predicted_class": u.get("classification", "unknown"),
                "understanding_score": u.get("score_index", 0.0),
                "confidence": u.get("confidence", 0.0),
                "probabilities": u.get("probabilities", {}),
                "display_name": u.get("display_name", ""),
            }
            actual_quiz_performance = top_topic.get("actual_quiz_performance") or {
                "status": "awaiting_quiz_submission"
            }
            c = top_topic.get("expressed_confidence", {}) or {}
            learning_confidence = {
                "expressed_confidence_level": c.get("classification", "unknown"),
                "expressed_confidence_score": c.get("score", 0.0),
                "status": c.get("status", "experimental_heuristic_baseline"),
            }
            o = top_topic.get("overall_performance", {}) or {}
            overall_score = o.get("overall_performance_score") if o.get("overall_performance_score") is not None else o.get("score")
            overall_topic_performance = {
                "overall_performance_score": overall_score,
                "status": o.get("status", "awaiting_quiz_submission"),
                "formula": o.get("calculation_formula") or o.get("formula", ""),
                "explanation": o.get("score_explanation") or o.get("explanation", ""),
            }
            discrepancy_analysis = {
                "discrepancy_detected": bool(o.get("discrepancies")),
                "details": "; ".join(o.get("discrepancies", [])) if o.get("discrepancies") else None,
            }
            conversational_signals = top_topic.get("conversational_signals", {}) or {}
            feature_explanation = top_topic.get("feature_explanation")
            pedagogical_feedback = u.get("pedagogical_feedback", "")
        else:
            topic_understanding = raw_json.get("topic_understanding", {}) or {}
            actual_quiz_performance = raw_json.get("actual_quiz_performance", {}) or {}
            learning_confidence = raw_json.get("learning_confidence", {}) or {}
            overall_topic_performance = raw_json.get("overall_topic_performance", {}) or {}
            discrepancy_analysis = raw_json.get("discrepancy_analysis", {}) or {}
            conversational_signals = raw_json.get("conversational_signals", {}) or {}
            feature_explanation = raw_json.get("feature_explanation")
            pedagogical_feedback = raw_json.get("pedagogical_feedback", "")

        return SessionAnalyticsResponse(
            session_id=session_id,
            user_id=user.user_id,
            subject=raw_sess.get("subject"),
            topic=raw_sess.get("topic"),
            source_agent=raw_sess.get("source_agent", "chatgpt"),
            status=raw_sess.get("status", "processed"),
            topic_understanding=topic_understanding,
            actual_quiz_performance=actual_quiz_performance,
            learning_confidence=learning_confidence,
            overall_topic_performance=overall_topic_performance,
            discrepancy_analysis=discrepancy_analysis,
            conversational_signals=conversational_signals,
            feature_explanation=feature_explanation,
            pedagogical_feedback=pedagogical_feedback,
            processed_at=analytics.get("processed_at", ""),
        )

    def get_student_performance(self, student_id: Optional[str], user: UserContext) -> StudentPerformanceResponse:
        """Retrieves aggregated performance for the authenticated student."""
        target_student_id = student_id or user.user_id
        if target_student_id != user.user_id:
            raise AuthorizationError(f"User '{user.user_id}' cannot access records for '{target_student_id}'.")

        summary = self.storage.get_student_summary(target_student_id)
        return StudentPerformanceResponse(**summary)

    def get_topic_analytics(self, topic: str, student_id: Optional[str], user: UserContext) -> TopicAnalyticsResponse:
        """Retrieves topic-specific analytics for the authenticated student."""
        target_student_id = student_id or user.user_id
        if target_student_id != user.user_id:
            raise AuthorizationError(f"User '{user.user_id}' cannot access records for '{target_student_id}'.")

        summary = self.storage.get_topic_summary(target_student_id, topic)
        if not summary:
            # Return empty structure
            return TopicAnalyticsResponse(
                topic=topic,
                subject=None,
                student_id=target_student_id,
                total_sessions=0,
                understanding_distribution={"understood": 0, "partial": 0, "confused": 0},
                average_understanding_score=0.0,
                quiz_performance={"average_score_percent": None, "quizzes_taken": 0},
                expressed_confidence={"average_confidence_score": 0.0},
                overall_topic_performance=None,
                historical_trend=[],
            )
        return TopicAnalyticsResponse(**summary)

    def list_learning_sessions(
        self,
        user: UserContext,
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> SessionListResponse:
        """Lists ingested sessions with pagination, strictly scoped to user."""
        total, items = self.storage.list_sessions(
            user_id=user.user_id,
            subject=subject,
            topic=topic,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            offset=offset,
        )
        session_items = [
            SessionListItem(
                session_id=it["session_id"],
                user_id=it["user_id"],
                subject=it.get("subject"),
                topic=it.get("topic"),
                source_agent=it.get("source_agent", "chatgpt"),
                status=it.get("status", "processed"),
                understanding_class=it.get("understanding_class"),
                overall_score=it.get("overall_score"),
                ingested_at=it.get("ingested_at", ""),
            )
            for it in items
        ]
        return SessionListResponse(
            total_count=total,
            limit=limit,
            offset=offset,
            sessions=session_items,
        )

    def submit_quiz_results(self, request: QuizSubmissionRequest, user: UserContext) -> QuizSubmissionResult:
        """
        Evaluates submitted quiz questions deterministically and updates overall topic score if linked to a session.
        """
        eval_result = evaluate_quiz_submission(
            questions=request.questions,
            answer_key=request.answer_key,
            student_answers=request.student_answers,
        )

        quiz_id = f"QZ_{uuid.uuid4().hex[:12].upper()}"

        effective_subject = request.subject.strip() if request.subject and request.subject.strip() else "Unspecified Subject"
        effective_topic = request.topic.strip() if request.topic and request.topic.strip() else "Unspecified Topic"

        quiz_storage_payload = {
            "quiz_id": quiz_id,
            "session_id": request.session_id,
            "user_id": user.user_id,
            "subject": effective_subject,
            "topic": effective_topic,
            "total_questions": eval_result.get("total_questions", 0),
            "correct_count": eval_result.get("correct_count", 0),
            "incorrect_count": eval_result.get("incorrect_count", 0),
            "unanswered_count": eval_result.get("unanswered_count", 0),
            "actual_score_percent": eval_result.get("actual_score_percent", 0.0),
            "question_breakdown": eval_result.get("question_breakdown") or eval_result.get("question_details", []),
            "topic_breakdown": eval_result.get("topic_breakdown", {}),
            "raw_submission": request.model_dump(),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        self.storage.save_quiz_submission(quiz_storage_payload)

        # If linked to a session, update the session analytics!
        updated_overall = None
        if request.session_id:
            existing_analytics = self.storage.get_analytics(request.session_id, user.user_id)
            if existing_analytics:
                full_json = existing_analytics["full_analytics_json"]
                
                # Check whether full_json uses multi-topic format or flat format
                if "topics" in full_json and full_json["topics"]:
                    top = full_json["topics"][0]
                    u_score = top.get("understanding", {}).get("score_index") or existing_analytics.get("understanding_score") or 0.0
                    c_score = top.get("expressed_confidence", {}).get("score") or existing_analytics.get("expressed_confidence_score") or 0.0
                else:
                    u_score = full_json.get("topic_understanding", {}).get("understanding_score") or existing_analytics.get("understanding_score") or 0.0
                    c_score = full_json.get("learning_confidence", {}).get("expressed_confidence_score") or existing_analytics.get("expressed_confidence_score") or 0.0
                
                q_score = eval_result["actual_score_percent"]

                scoring_engine = TopicPerformanceScoringEngine()
                new_overall = scoring_engine.compute_topic_performance(
                    topic=effective_topic,
                    understanding_input=u_score,
                    actual_quiz_score=q_score,
                    expressed_confidence_score=c_score,
                    session_id=request.session_id,
                    student_id=user.user_id,
                )

                quiz_perf_dict = {
                    "status": "evaluated",
                    "actual_score_percent": q_score,
                    "correct_count": eval_result["correct_count"],
                    "total_questions": eval_result["total_questions"],
                    "incorrect_count": eval_result["incorrect_count"],
                    "unanswered_count": eval_result["unanswered_count"],
                }

                if "topics" in full_json and full_json["topics"]:
                    full_json["topics"][0]["actual_quiz_performance"] = quiz_perf_dict
                    full_json["topics"][0]["overall_performance"] = new_overall
                
                full_json["actual_quiz_performance"] = quiz_perf_dict
                full_json["overall_topic_performance"] = new_overall
                
                self.storage.save_analytics(request.session_id, user.user_id, full_json)
                updated_overall = new_overall

        return QuizSubmissionResult(
            quiz_id=quiz_id,
            session_id=request.session_id,
            user_id=user.user_id,
            total_questions=eval_result.get("total_questions", 0),
            correct_count=eval_result.get("correct_count", 0),
            incorrect_count=eval_result.get("incorrect_count", 0),
            unanswered_count=eval_result.get("unanswered_count", 0),
            actual_score_percent=eval_result.get("actual_score_percent", 0.0),
            question_breakdown=eval_result.get("question_breakdown", []),
            topic_breakdown=eval_result.get("topic_breakdown", {}),
            evaluated_at=quiz_storage_payload["evaluated_at"],
            updated_overall_performance=updated_overall,
        )

    def retry_failed_session(self, session_id: str, user: UserContext) -> SessionIngestResult:
        """
        Retries ML analytics processing for an existing raw session without creating duplicate raw records.
        """
        raw = self.storage.get_raw_session(session_id, user.user_id)
        if not raw:
            raise KeyError(f"Session '{session_id}' not found for user '{user.user_id}'.")

        raw_json = raw.get("raw_json", {})
        messages = raw_json.get("messages", [])
        quiz_submission = raw_json.get("quiz_submission")

        req = SessionIngestRequest(
            messages=[MessageItem(**m) for m in messages],
            subject=raw.get("subject"),
            topic=raw.get("topic"),
            session_id=session_id,
            student_id=user.user_id,
            source_agent=raw.get("source_agent", "chatgpt"),
            metadata=raw_json.get("metadata", {}),
            quiz_submission=quiz_submission,
        )
        return self.ingest_session(req, user)



# Global ingestion service singleton
_ingestion_service: Optional[IngestionService] = None


def get_ingestion_service() -> IngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service
