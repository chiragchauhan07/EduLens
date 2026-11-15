"""
scoring_engine.py — Transparent Topic-Wise Performance Scoring Engine for EduLens.

Combines four distinct performance dimensions into an interpretable overall evaluation:
1. Topic Understanding Score (from ML Classification / calibrated probabilities)
2. Actual Quiz Score (from deterministic quiz answer evaluation)
3. Expressed Learning Confidence Score (from conversational marker estimation)
4. (Optional) Predicted Quiz Score comparison (from conversational regression)

FORMULA:
Overall Topic Performance =
    (Understanding Score × w_u) + (Actual Quiz Score × w_q) + (Expressed Confidence Score × w_c)
Default provisional weights: w_u = 0.40, w_q = 0.40, w_c = 0.20 (Sum = 1.0)
"""

from typing import Any, Dict, Optional, Tuple, Union

DEFAULT_WEIGHTS = {
    "understanding": 0.40,
    "actual_quiz": 0.40,
    "confidence": 0.20,
}

# Explicit documented numerical conversion from categorical Bloom understanding level
UNDERSTANDING_CLASS_MAP = {
    "understood": 90.0,
    "partial": 55.0,
    "confused": 20.0,
}


def convert_understanding_to_score(
    classification: Optional[str] = None,
    probabilities: Optional[Dict[str, float]] = None,
    explicit_score: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Converts understanding classification to a transparent 0-100 score index.
    Does NOT claim to be a psychometric measurement; documented as an expected value mapping.
    """
    if explicit_score is not None:
        return float(np_clip(explicit_score, 0.0, 100.0)), "explicit_score_provided"

    if probabilities and len(probabilities) == 3:
        # Expected value formula based on calibrated class probabilities
        p_und = float(probabilities.get("understood", 0.0))
        p_part = float(probabilities.get("partial", 0.0))
        p_conf = float(probabilities.get("confused", 0.0))
        total_p = p_und + p_part + p_conf
        if total_p > 0:
            expected_score = (p_und * 100.0) + (p_part * 60.0) + (p_conf * 20.0)
            return round(expected_score / total_p, 1), "probability_weighted_expectation"

    if classification:
        norm_cls = classification.strip().lower()
        score = UNDERSTANDING_CLASS_MAP.get(norm_cls, 50.0)
        return score, "categorical_discrete_mapping"

    return 50.0, "default_neutral_baseline"


def np_clip(val: float, low: float, high: float) -> float:
    return max(low, min(val, high))


class TopicPerformanceScoringEngine:
    """
    Deterministic, configurable composite performance engine.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights if weights is not None else dict(DEFAULT_WEIGHTS)
        self._validate_weights()

    def _validate_weights(self):
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Scoring weights must sum to 1.0 (current sum: {total:.4f})")

    def compute_topic_performance(
        self,
        topic: str,
        understanding_input: Union[str, float, Dict[str, Any]],
        actual_quiz_score: Optional[float] = None,
        expressed_confidence_score: Optional[float] = None,
        predicted_quiz_score: Optional[float] = None,
        session_id: Optional[str] = None,
        student_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Computes overall topic performance across dimensions.
        """
        # 1. Resolve Understanding Score
        u_score = None
        u_method = "none"
        u_class = "unknown"

        if isinstance(understanding_input, dict):
            u_class = understanding_input.get("classification", understanding_input.get("predicted_class", "unknown"))
            u_probs = understanding_input.get("probabilities")
            u_exp = understanding_input.get("score")
            u_score, u_method = convert_understanding_to_score(u_class, u_probs, u_exp)
        elif isinstance(understanding_input, (int, float)):
            u_score = float(np_clip(understanding_input, 0.0, 100.0))
            u_method = "direct_numerical"
            u_class = "understood" if u_score >= 75 else ("partial" if u_score >= 45 else "confused")
        elif isinstance(understanding_input, str):
            u_class = understanding_input
            u_score, u_method = convert_understanding_to_score(understanding_input)

        # 2. Resolve Confidence Score
        c_score = None
        if expressed_confidence_score is not None:
            c_score = float(np_clip(expressed_confidence_score, 0.0, 100.0))

        # 3. Resolve Actual Quiz Score & Discrepancies
        q_score = None
        if actual_quiz_score is not None:
            q_score = float(np_clip(actual_quiz_score, 0.0, 100.0))

        # 4. Handle Missing Quiz Scenario
        if q_score is None:
            # Quiz not yet submitted — DO NOT FABRICATE A SCORE!
            interim_score = None
            interim_formula = None
            if u_score is not None and c_score is not None:
                # Optional provisional score rescaling u (67%) and c (33%)
                w_u_norm = self.weights["understanding"] / (self.weights["understanding"] + self.weights["confidence"])
                w_c_norm = self.weights["confidence"] / (self.weights["understanding"] + self.weights["confidence"])
                interim_score = round((u_score * w_u_norm) + (c_score * w_c_norm), 1)
                interim_formula = f"({u_score} × {w_u_norm:.2f}) + ({c_score} × {w_c_norm:.2f}) = {interim_score}/100"

            explanation = (
                "Actual quiz assessment has not been submitted for this topic. "
                "Overall topic performance requires verified quiz evaluation to avoid misleading composite scores."
            )

            return {
                "topic": topic,
                "session_id": session_id,
                "student_id": student_id,
                "overall_performance_score": None,
                "status": "awaiting_quiz_submission",
                "provisional_conversational_score": interim_score,
                "provisional_formula": interim_formula,
                "dimension_scores": {
                    "topic_understanding": {
                        "score": u_score,
                        "classification": u_class,
                        "conversion_method": u_method,
                        "weight": self.weights["understanding"],
                    },
                    "actual_quiz_performance": {
                        "score": None,
                        "status": "not_submitted",
                        "weight": self.weights["actual_quiz"],
                    },
                    "expressed_confidence": {
                        "score": c_score,
                        "weight": self.weights["confidence"],
                    },
                    "predicted_quiz_performance": {
                        "score": round(predicted_quiz_score, 1) if predicted_quiz_score is not None else None,
                        "type": "conversational_regression_estimate",
                    },
                },
                "score_explanation": explanation,
                "weights_used": dict(self.weights),
            }

        # 5. Full Evaluation (All required dimensions present)
        # Default c_score to 50.0 if not measured
        eff_c_score = c_score if c_score is not None else 50.0

        w_u = self.weights["understanding"]
        w_q = self.weights["actual_quiz"]
        w_c = self.weights["confidence"]

        overall = round((u_score * w_u) + (q_score * w_q) + (eff_c_score * w_c), 1)
        formula_str = (
            f"({u_score:.1f} × {w_u:.2f}) + ({q_score:.1f} × {w_q:.2f}) + ({eff_c_score:.1f} × {w_c:.2f}) = {overall:.1f}/100"
        )

        # Discrepancy Analysis
        discrepancies = []
        if u_score >= 75.0 and q_score < 50.0:
            discrepancies.append(
                "High conversational understanding was demonstrated, but quiz assessment score was low. "
                "May indicate surface fluency or recall gaps under testing conditions."
            )
        elif u_score < 50.0 and q_score >= 80.0:
            discrepancies.append(
                "Student expressed conversational confusion during the session, but achieved a high quiz score. "
                "Indicates productive struggle or self-resolution before the assessment."
            )

        delta_pred = None
        if predicted_quiz_score is not None:
            delta_pred = round(q_score - predicted_quiz_score, 1)

        return {
            "topic": topic,
            "session_id": session_id,
            "student_id": student_id,
            "overall_performance_score": overall,
            "status": "complete",
            "calculation_formula": formula_str,
            "dimension_scores": {
                "topic_understanding": {
                    "score": u_score,
                    "classification": u_class,
                    "conversion_method": u_method,
                    "weight": w_u,
                },
                "actual_quiz_performance": {
                    "score": q_score,
                    "status": "evaluated",
                    "weight": w_q,
                },
                "expressed_confidence": {
                    "score": c_score,
                    "weight": w_c,
                },
                "predicted_quiz_performance": {
                    "score": round(predicted_quiz_score, 1) if predicted_quiz_score is not None else None,
                    "type": "conversational_regression_estimate",
                    "actual_minus_predicted_delta": delta_pred,
                },
            },
            "discrepancy_insights": discrepancies,
            "score_explanation": (
                f"Overall Topic Performance is {overall}/100 calculated using weighted composite: "
                f"40% Understanding ({u_score}), 40% Actual Quiz ({q_score}), 20% Expressed Confidence ({eff_c_score})."
            ),
            "weights_used": dict(self.weights),
        }
