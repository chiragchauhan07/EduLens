"""
unified_pipeline.py — Unified End-to-End Learning Intelligence Pipeline for EduLens.

Coordinates the complete 4-dimension evaluation workflow:
1. Ingestion & DSP Validation
2. Conversational NLP & Feature Extraction
3. Topic Understanding Classification (LR, SVM, MNB)
4. Expressed Confidence Estimation
5. Actual Quiz Evaluation & Conversational Quiz Prediction
6. Topic-Wise Performance Aggregation via Scoring Engine

Produces a clean, structured, integration-ready payload for MCP and frontend dashboards.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml_prototype.dsp import (
    validate_session_json,
    clean_and_normalize_session,
    extract_dsp_conversation_metrics,
)
from ml_prototype.nlp import parse_and_clean_conversation
from ml_prototype.features import extract_conversation_stats
from ml_prototype.predict import EduLensPredictor
from ml_prototype.confidence import ExpressedConfidenceEstimator
from ml_prototype.quiz_evaluator import evaluate_quiz_submission
from ml_prototype.scoring_engine import TopicPerformanceScoringEngine


class EduLensUnifiedPipeline:
    """
    End-to-end learning intelligence orchestrator for student learning conversations.
    """

    def __init__(
        self,
        algorithm: str = "logistic_regression",
        scoring_weights: Optional[Dict[str, float]] = None,
        load_quiz_regressor: bool = True,
    ):
        self.algorithm = algorithm
        self.predictor = EduLensPredictor(algorithm=algorithm, load_quiz_regressor=load_quiz_regressor)
        self.confidence_estimator = ExpressedConfidenceEstimator()
        self.scoring_engine = TopicPerformanceScoringEngine(weights=scoring_weights)

    def process_session(
        self,
        session_input: Any,
        quiz_submission: Optional[Dict[str, Any]] = None,
        explain: bool = False,
        topic_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes the full EduLens learning analytics pipeline on a student conversation.

        Parameters:
        - session_input: Raw JSON session dict, turn list, or plain text transcript.
        - quiz_submission: Optional dict containing actual quiz questions, answer key, and student answers.
        - explain: Whether to include feature attribution drivers.
        - topic_override: Explicit topic name if not present in session metadata.

        Returns structured analytics dictionary across all 4 performance dimensions.
        """
        # 1. DSP Validation
        is_valid, val_errors, sanitized = validate_session_json(session_input)
        if not is_valid:
            return {
                "status": "validation_failed",
                "errors": val_errors,
                "session_id": "UNKNOWN",
                "student_id": "UNKNOWN",
                "topics": [],
                "dsp_metrics": {},
                "summary": "Session validation failed before processing.",
            }

        # 2. Check for Multi-Topic Structure
        # If the input explicitly has multiple topic blocks, e.g. 'topics': [{"topic": "T1", "messages": [...]}, ...]
        if isinstance(sanitized, dict) and "topics" in sanitized and isinstance(sanitized["topics"], list):
            multi_topic_results = []
            for t_item in sanitized["topics"]:
                t_name = t_item.get("topic", "General Topic")
                t_msgs = t_item.get("student_messages", t_item.get("messages", t_item.get("turns", [])))
                t_quiz = t_item.get("quiz_submission", quiz_submission)

                t_res = self._process_single_topic(
                    topic_name=t_name,
                    conversation_data=t_msgs,
                    quiz_submission=t_quiz,
                    explain=explain,
                    session_id=sanitized.get("chat_id", "MULTI_TOPIC_SESSION"),
                    student_id=sanitized.get("student_id", "STUDENT"),
                    subject=sanitized.get("subject", "Computer Science"),
                )
                multi_topic_results.append(t_res)

            dsp_stats = extract_dsp_conversation_metrics(sanitized)
            return {
                "status": "success",
                "session_id": sanitized.get("chat_id", "MULTI_TOPIC_SESSION"),
                "student_id": sanitized.get("student_id", "STUDENT"),
                "subject": sanitized.get("subject", "Computer Science"),
                "is_multi_topic": True,
                "topics": multi_topic_results,
                "dsp_metrics": dsp_stats,
            }

        # 3. Standard Single-Topic Processing
        normalized = clean_and_normalize_session(sanitized)
        topic_name = topic_override or normalized.get("topic") or "General Topic"
        session_id = normalized.get("chat_id", "SESSION")
        student_id = normalized.get("student_id", "STUDENT")
        subject = normalized.get("subject", "Computer Science")

        # If session object has embedded quiz data and no separate quiz_submission was passed
        if quiz_submission is None and normalized.get("raw_quiz_data"):
            # Check if actual questions are available or just score summary
            raw_q = normalized["raw_quiz_data"]
            if "questions" in raw_q or "answer_key" in raw_q:
                quiz_submission = raw_q

        topic_result = self._process_single_topic(
            topic_name=topic_name,
            conversation_data=normalized["student_messages"],
            quiz_submission=quiz_submission,
            explain=explain,
            session_id=session_id,
            student_id=student_id,
            subject=subject,
        )

        return {
            "status": "success",
            "session_id": session_id,
            "student_id": student_id,
            "subject": subject,
            "is_multi_topic": False,
            "topics": [topic_result],
            "dsp_metrics": normalized["dsp_metrics"],
        }

    def _process_single_topic(
        self,
        topic_name: str,
        conversation_data: Any,
        quiz_submission: Optional[Dict[str, Any]],
        explain: bool,
        session_id: str,
        student_id: str,
        subject: str,
    ) -> Dict[str, Any]:
        """Processes a single topic conversation through the 4-dimension pipeline."""
        # 1. NLP Text Parsing
        student_msgs, teacher_msgs, full_text = parse_and_clean_conversation(conversation_data)

        if not student_msgs:
            return {
                "topic": topic_name,
                "status": "empty_conversation",
                "error": "No student utterances found for topic.",
                "understanding": None,
                "actual_quiz_performance": None,
                "expressed_confidence": None,
                "overall_performance": None,
            }

        # 2. Dimension 1: Topic Understanding Classification
        pred_res = self.predictor.predict(student_msgs, explain=explain)
        u_class = pred_res.get("predicted_class", "unknown")
        u_probs = pred_res.get("probabilities", {})
        u_conf = pred_res.get("confidence", 0.0)
        feedback = pred_res.get("pedagogical_feedback", "")
        pred_quiz_score = pred_res.get("predicted_quiz_score")
        feature_exp = pred_res.get("feature_explanation")

        # 3. Dimension 2: Actual Quiz Performance (Deterministic)
        actual_quiz_res = None
        actual_quiz_score = None

        if quiz_submission is not None:
            actual_quiz_res = evaluate_quiz_submission(submission_payload=quiz_submission)
            if "actual_score_percent" in actual_quiz_res and actual_quiz_res["actual_score_percent"] is not None:
                actual_quiz_score = actual_quiz_res["actual_score_percent"]

        # 4. Dimension 3: Expressed Learning Confidence (New Model)
        conf_res = self.confidence_estimator.predict(student_msgs)

        # 5. Dimension 4: Overall Topic Performance (Composite Scoring Engine)
        scoring_res = self.scoring_engine.compute_topic_performance(
            topic=topic_name,
            understanding_input={
                "classification": u_class,
                "probabilities": u_probs,
            },
            actual_quiz_score=actual_quiz_score,
            expressed_confidence_score=conf_res.get("score"),
            predicted_quiz_score=pred_quiz_score,
            session_id=session_id,
            student_id=student_id,
        )

        return {
            "topic": topic_name,
            "subject": subject,
            "session_id": session_id,
            "student_id": student_id,
            "understanding": {
                "classification": u_class,
                "display_name": pred_res.get("display_name", u_class.capitalize()),
                "confidence": u_conf,
                "probabilities": u_probs,
                "model_used": self.algorithm,
                "pedagogical_feedback": feedback,
                "score_index": scoring_res["dimension_scores"]["topic_understanding"]["score"],
            },
            "actual_quiz_performance": actual_quiz_res,
            "expressed_confidence": {
                "classification": conf_res.get("classification"),
                "display_name": conf_res.get("display_name"),
                "score": conf_res.get("score"),
                "signals": conf_res.get("confidence_signals"),
                "explanation": conf_res.get("explanation"),
                "status": conf_res.get("status"),
            },
            "predicted_quiz_performance": {
                "score": pred_quiz_score,
                "model_type": "RandomForest / Ridge Regressor",
                "delta_vs_actual": (
                    round(actual_quiz_score - pred_quiz_score, 1)
                    if actual_quiz_score is not None and pred_quiz_score is not None
                    else None
                ),
            },
            "overall_performance": {
                "score": scoring_res.get("overall_performance_score"),
                "status": scoring_res.get("status"),
                "formula": scoring_res.get("calculation_formula") or scoring_res.get("provisional_formula"),
                "explanation": scoring_res.get("score_explanation"),
                "discrepancies": scoring_res.get("discrepancy_insights", []),
            },
            "conversational_signals": pred_res.get("conversational_signals", {}),
            "feature_explanation": feature_exp,
        }
