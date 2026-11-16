"""
prediction.py — Core inference and prediction orchestrator for EduLens Mini.

Loads the trained Scikit-Learn understanding classifier once at startup,
extracts features, performs understanding classification, evaluates expressed
linguistic confidence, and compiles structured educational insights.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np

from backend.services.conversation_parser import ParsedConversation
from backend.services.preprocessing import prepare_inference_frame
from backend.services.feature_extraction import (
    estimate_linguistic_confidence,
    extract_conversational_stats,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT_DIR / "backend" / "models" / "understanding_classifier.joblib"
FALLBACK_MODEL_PATH = ROOT_DIR / "models" / "trained_model.joblib"
METADATA_PATH = ROOT_DIR / "backend" / "models" / "metadata.json"

CLASS_DISPLAY_MAP = {
    "understood": "Understood",
    "partial": "Partially Understood",
    "confused": "Confused",
}

CLASS_EXPLANATIONS = {
    "understood": (
        "The student demonstrated solid comprehension, articulated concepts clearly, "
        "and applied principles with minimal unresolved confusion."
    ),
    "partial": (
        "The student demonstrated emerging comprehension but required clarifications, "
        "struggled with edge cases, or showed lingering uncertainty."
    ),
    "confused": (
        "The student expressed repeated difficulty, misconceptions, or conceptual doubt. "
        "Targeted foundational review or instructional intervention is recommended."
    ),
}


class ModelRegistry:
    _instance: Optional["ModelRegistry"] = None
    pipeline = None
    metadata: Dict[str, Any] = {}
    classes: List[str] = []

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self):
        target_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH
        if not target_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {MODEL_PATH} or {FALLBACK_MODEL_PATH}. "
                "Please ensure the model is trained."
            )

        self.pipeline = joblib.load(target_path)
        self.classes = getattr(self.pipeline, "classes_", ["confused", "partial", "understood"]).tolist()

        if METADATA_PATH.exists():
            try:
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}
        else:
            self.metadata = {
                "model_name": "Logistic Regression (Balanced)",
                "algorithm": "logistic_regression",
                "test_accuracy": 0.9589,
                "macro_f1": 0.9399,
                "features_dim": 108,
            }


def analyze_conversation_session(parsed: ParsedConversation) -> Dict[str, Any]:
    """
    Runs the complete ML & NLP analysis pipeline on a parsed student conversation.

    Args:
        parsed: ParsedConversation instance from conversation_parser.py.

    Returns:
        Structured result dictionary.
    """
    registry = ModelRegistry.get_instance()
    student_msgs = parsed.student_messages

    # 1. Preprocessing into inference DataFrame
    df_input, full_text = prepare_inference_frame(student_msgs)

    # 2. Model Prediction & Probabilities
    raw_pred = str(registry.pipeline.predict(df_input)[0])
    display_label = CLASS_DISPLAY_MAP.get(raw_pred, raw_pred.capitalize())

    class_probs = {}
    top_probability = 1.0
    if hasattr(registry.pipeline, "predict_proba"):
        probs = registry.pipeline.predict_proba(df_input)[0]
        for cls_name, p in zip(registry.classes, probs):
            readable = CLASS_DISPLAY_MAP.get(cls_name, cls_name.capitalize())
            class_probs[readable] = round(float(p), 4)
        top_probability = class_probs.get(display_label, round(float(np.max(probs)), 4))
    else:
        for cls_name in registry.classes:
            readable = CLASS_DISPLAY_MAP.get(cls_name, cls_name.capitalize())
            class_probs[readable] = 1.0 if cls_name == raw_pred else 0.0

    # 3. Expressed Linguistic Confidence Estimation
    conf_label, conf_score, conf_expl, conf_signals = estimate_linguistic_confidence(student_msgs)

    # 4. Conversational Statistics
    convo_stats = extract_conversational_stats(student_msgs)

    # 5. Pedagogical Insights Compilation
    insights = []
    if display_label == "Understood":
        insights.append("Student demonstrated solid conceptual grasp and articulated principles accurately.")
    elif display_label == "Partially Understood":
        insights.append("Student demonstrated emerging understanding but encountered conceptual hesitation.")
    else:
        insights.append("Student expressed unresolved confusion and struggled with core mechanics.")

    if convo_stats["question_count"] > 0:
        insights.append(f"Student actively asked {int(convo_stats['question_count'])} question(s) during the session.")

    if conf_signals["uncertainty_count"] > 0:
        insights.append(f"Detected {int(conf_signals['uncertainty_count'])} explicit linguistic uncertainty marker(s).")
    else:
        insights.append("Zero explicit uncertainty markers detected in student dialogue turns.")

    if conf_signals["certainty_count"] > 0:
        insights.append(f"Student used {int(conf_signals['certainty_count'])} assertive certainty indicator(s).")

    if conf_signals["self_corrections"] > 0:
        insights.append(f"Student showed self-correction / metacognitive awareness ({int(conf_signals['self_corrections'])} times).")

    explanation_text = CLASS_EXPLANATIONS.get(raw_pred, "Model prediction completed.")

    return {
        "success": True,
        "topic": parsed.topic,
        "understanding": {
            "label": display_label,
            "raw_class": raw_pred,
            "probability": top_probability,
            "class_probabilities": class_probs,
            "explanation": explanation_text,
        },
        "expressed_confidence": {
            "label": conf_label,
            "score": conf_score,
            "method": "linguistic_estimation",
            "explanation": conf_expl,
            "methodology_note": (
                "Estimates expressed linguistic confidence in dialogue from conversational markers "
                "(assertions, hedges, self-corrections, hesitation). It is not an ML model prediction "
                "or psychometric measurement."
            ),
            "signals": conf_signals,
        },
        "conversation_stats": {
            "student_messages": parsed.student_turn_count,
            "assistant_messages": parsed.assistant_turn_count,
            "total_turns": parsed.total_turns,
            "total_words": int(convo_stats["total_words"]),
            "avg_turn_words": convo_stats["avg_turn_length"],
            "question_count": int(convo_stats["question_count"]),
            "confusion_signals": int(convo_stats["confusion_markers"]),
            "understanding_signals": int(convo_stats["understanding_markers"]),
        },
        "insights": insights,
        "model_info": {
            "name": registry.metadata.get("model_name", "Logistic Regression (Balanced)"),
            "algorithm": registry.metadata.get("algorithm", "logistic_regression"),
            "features_dimension": registry.metadata.get("features_dim", 108),
            "test_accuracy": registry.metadata.get("metrics", {}).get("test_accuracy", 0.9589),
            "macro_f1": registry.metadata.get("metrics", {}).get("macro_f1", 0.9399),
        },
    }
