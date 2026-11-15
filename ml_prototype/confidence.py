"""
confidence.py — Expressed Learning Confidence Estimator for EduLens.

Estimates how confident a student appears while learning, discussing, and answering
questions about a topic.

IMPORTANT METHODOLOGICAL DISTINCTION:
This component estimates *expressed linguistic confidence* derived from conversational
markers (assertions, hedges, self-corrections, hesitation fillers). It does NOT purport
to measure internal psychological conviction or psychometric self-efficacy.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# TARGET CATEGORIES & METADATA
# ─────────────────────────────────────────────────────────────────────────────
CONFIDENCE_LEVELS = ["low", "moderate", "high"]

CONFIDENCE_METADATA = {
    "low": {
        "display_name": "Low Expressed Confidence",
        "description": "Student frequently expresses doubt, uses hedges, hesitates, or repeatedly seeks reassurance.",
        "badge_color": "red",
    },
    "moderate": {
        "display_name": "Moderate Expressed Confidence",
        "description": "Student communicates with balanced inquiry, occasional hesitation, and tentative explanations.",
        "badge_color": "orange",
    },
    "high": {
        "display_name": "High Expressed Confidence",
        "description": "Student articulates claims assertively, offers independent explanations, and exhibits minimal hedging.",
        "badge_color": "green",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# REGEX PATTERNS FOR EXPRESSED CONFIDENCE SIGNALS
# ─────────────────────────────────────────────────────────────────────────────
RE_UNCERTAINTY = re.compile(
    r"\b(not sure|maybe|i think|might be|i guess|guess|probably|doubt|unclear|hesitant|"
    r"idk|i don'?t know|not very confident|i'?m blank|no idea|could it be|is it possible)\b",
    re.IGNORECASE,
)

RE_CERTAINTY = re.compile(
    r"\b(definitely|obviously|clearly|i am certain|i'?m certain|i know|always|guaranteed|"
    r"confirmed|of course|without doubt|precisely|surely|positive that)\b",
    re.IGNORECASE,
)

RE_HESITATION = re.compile(
    r"(\b(uh|um|hmm|er|well\.\.\.)\b|\.{3,})",
    re.IGNORECASE,
)

RE_CONFIRMATION_REQUEST = re.compile(
    r"\b(am i right|is that right|is this correct|right\?|correct\?|does that make sense|"
    r"am i on the right track|is it\?|did i get that right)\b",
    re.IGNORECASE,
)

RE_SELF_CORRECTION = re.compile(
    r"\b(oh wait|no wait|i mean|actually it'?s|my mistake|i meant|wait let me rephrase|"
    r"sorry i meant|scratch that)\b",
    re.IGNORECASE,
)

RE_INDEPENDENT_EXPLANATION = re.compile(
    r"\b(so basically|the reason is|because|which means that|therefore|it works by|"
    r"if we look at|for instance|in other words)\b",
    re.IGNORECASE,
)

RE_HINT_DEPENDENCE = re.compile(
    r"\b(give me a hint|tell me the answer|just give the code|what is the answer|"
    r"can'?t solve this|just tell me|show me the solution)\b",
    re.IGNORECASE,
)


def extract_confidence_signals(student_messages: List[str]) -> Dict[str, float]:
    """
    Extracts numerical conversational signals indicative of expressed confidence.
    """
    if not student_messages:
        return {
            "uncertainty_count": 0.0,
            "certainty_count": 0.0,
            "hesitation_count": 0.0,
            "confirmation_requests": 0.0,
            "self_corrections": 0.0,
            "independent_explanations": 0.0,
            "hint_dependencies": 0.0,
            "total_student_turns": 0.0,
            "total_words": 0.0,
            "avg_turn_words": 0.0,
            "assertion_hedge_ratio": 0.0,
        }

    full_text = " ".join(student_messages)
    words = [w for w in full_text.split() if w.strip()]
    n_turns = len(student_messages)
    n_words = len(words)

    unc_count = float(len(RE_UNCERTAINTY.findall(full_text)))
    cert_count = float(len(RE_CERTAINTY.findall(full_text)))
    hes_count = float(len(RE_HESITATION.findall(full_text)))
    conf_req_count = float(len(RE_CONFIRMATION_REQUEST.findall(full_text)))
    self_corr_count = float(len(RE_SELF_CORRECTION.findall(full_text)))
    indep_expl_count = float(len(RE_INDEPENDENT_EXPLANATION.findall(full_text)))
    hint_dep_count = float(len(RE_HINT_DEPENDENCE.findall(full_text)))

    hedge_sum = unc_count + hes_count + (0.5 * conf_req_count)
    assert_sum = cert_count + indep_expl_count + (0.5 * self_corr_count)
    assertion_hedge_ratio = float((assert_sum + 1.0) / (hedge_sum + 1.0))

    return {
        "uncertainty_count": unc_count,
        "certainty_count": cert_count,
        "hesitation_count": hes_count,
        "confirmation_requests": conf_req_count,
        "self_corrections": self_corr_count,
        "independent_explanations": indep_expl_count,
        "hint_dependencies": hint_dep_count,
        "total_student_turns": float(n_turns),
        "total_words": float(n_words),
        "avg_turn_words": float(n_words / n_turns) if n_turns else 0.0,
        "assertion_hedge_ratio": round(assertion_hedge_ratio, 3),
    }


def compute_expressed_confidence_score(signals: Dict[str, float]) -> Tuple[str, float, str]:
    """
    Computes an expressed learning confidence score between 0.0 and 100.0
    and returns (category, score, explanation).

    Formula Rationale:
    - Base score = 55.0 (neutral baseline for student dialogue)
    - Positive boosters: certainty (+7.0 each), independent explanations (+6.0 each),
      self-correction (+4.0 each, reflects metacognitive awareness)
    - Dampeners: uncertainty markers (-8.0 each), hesitation markers (-5.0 each),
      confirmation requests (-4.0 each), hint dependencies (-9.0 each)
    - Scaled and bounded to [5.0, 95.0].
    """
    if signals.get("total_student_turns", 0.0) == 0:
        return "low", 0.0, "No student utterances provided."

    score = 55.0

    # Boosters
    score += signals.get("certainty_count", 0.0) * 7.0
    score += signals.get("independent_explanations", 0.0) * 6.0
    score += signals.get("self_corrections", 0.0) * 4.0

    # Dampeners
    score -= signals.get("uncertainty_count", 0.0) * 8.0
    score -= signals.get("hesitation_count", 0.0) * 5.0
    score -= signals.get("confirmation_requests", 0.0) * 4.0
    score -= signals.get("hint_dependencies", 0.0) * 9.0

    # Slight adjustment based on turn length (very short turns slightly depress confidence)
    if signals.get("avg_turn_words", 0.0) < 6.0:
        score -= 5.0
    elif signals.get("avg_turn_words", 0.0) > 18.0:
        score += 4.0

    # Clamp score
    clamped_score = round(float(np.clip(score, 5.0, 95.0)), 1)

    if clamped_score < 40.0:
        level = "low"
        explanation = (
            f"Expressed confidence is low ({clamped_score}/100) due to frequent expressions of doubt "
            f"({int(signals['uncertainty_count'])}) or repeated hesitation."
        )
    elif clamped_score <= 70.0:
        level = "moderate"
        explanation = (
            f"Expressed confidence is moderate ({clamped_score}/100) reflecting balanced inquiry with "
            f"some tentative phrasing or confirmation checks."
        )
    else:
        level = "high"
        explanation = (
            f"Expressed confidence is high ({clamped_score}/100) marked by assertive statements "
            f"({int(signals['certainty_count'])}) and independent explanation formulations."
        )

    return level, clamped_score, explanation


class ExpressedConfidenceEstimator:
    """
    Inference interface for Expressed Learning Confidence Estimation.
    Combines rule-calibrated conversational signal scoring with optional trained classifier.
    """

    def __init__(self, model_artifact_path: Optional[Union[str, Path]] = None):
        self.artifact_path = (
            Path(model_artifact_path)
            if model_artifact_path
            else ROOT_DIR / "models" / "confidence" / "model.joblib"
        )
        self.trained_pipeline = None
        if self.artifact_path.exists():
            try:
                self.trained_pipeline = joblib.load(self.artifact_path)
            except Exception as e:
                print(f"Note: Could not load confidence model artifact ({e}), using calibrated signal baseline.")

    def predict(self, conversation_input: Union[Dict[str, Any], List[str], str]) -> Dict[str, Any]:
        """
        Estimates expressed learning confidence for a conversation session.
        """
        from ml_prototype.data_loader import extract_student_messages

        if isinstance(conversation_input, dict):
            student_msgs = extract_student_messages(conversation_input)
        elif isinstance(conversation_input, list):
            student_msgs = [str(m).strip() for m in conversation_input if str(m).strip()]
        elif isinstance(conversation_input, str):
            student_msgs = [l.strip() for l in conversation_input.split("\n") if l.strip()]
        else:
            student_msgs = []

        if not student_msgs:
            return {
                "classification": "unknown",
                "display_name": "Unknown",
                "score": None,
                "confidence_signals": {},
                "explanation": "Unable to estimate confidence from empty conversation.",
                "status": "insufficient_data",
                "is_validated": False,
            }

        signals = extract_confidence_signals(student_msgs)
        category, score, explanation = compute_expressed_confidence_score(signals)

        meta = CONFIDENCE_METADATA.get(category, {
            "display_name": category.capitalize(),
            "badge_color": "gray",
        })

        return {
            "classification": category,
            "display_name": meta["display_name"],
            "score": score,
            "confidence_signals": signals,
            "explanation": explanation,
            "status": "experimental_heuristic_baseline",
            "is_validated": False,
            "methodology_note": (
                "Estimates expressed linguistic confidence in dialogue. Does not measure "
                "actual internal psychometric confidence or mastery."
            ),
        }


def audit_available_confidence_datasets() -> Dict[str, Any]:
    """
    Audits all raw repository datasets to report available confidence labels and explain the data gap.
    """
    findings = {}

    # Primary dataset
    primary_path = ROOT_DIR / "data" / "raw" / "edulens_chats_dataset.json"
    if primary_path.exists():
        with open(primary_path, "r", encoding="utf-8") as f:
            p_data = json.load(f)
        findings["primary_dataset"] = {
            "file": primary_path.name,
            "records": len(p_data),
            "confidence_labels_present": False,
            "reason": "Legacy schema v1.0 did not annotate student confidence.",
        }

    # Pilot v2.1 dataset
    pilot_path = ROOT_DIR / "data" / "raw" / "edulens_expanded_pilot_020_v2.json"
    if pilot_path.exists():
        with open(pilot_path, "r", encoding="utf-8") as f:
            pilot_data = json.load(f)
        conf_values = [d.get("student_confidence") for d in pilot_data if "student_confidence" in d]
        non_unknown = [v for v in conf_values if v != "unknown"]
        counts = pd.Series(conf_values).value_counts().to_dict()
        distribution = {str(k): int(v) for k, v in counts.items()}
        findings["pilot_v2_1"] = {
            "file": pilot_path.name,
            "records": len(pilot_data),
            "confidence_labels_present": True,
            "distribution": distribution,
            "usable_annotated_records": len(non_unknown),
            "reason": (
                "Schema v2.1 restricts confidence annotations strictly to turns with explicit "
                "self-expressions of certainty or doubt; 10/20 sessions were annotated as 'unknown'."
            ),
        }

    findings["gap_analysis"] = (
        "No large-scale human-validated student confidence dataset currently exists. "
        "The primary 430-session dataset contains zero confidence labels. "
        "The pilot dataset contains only 10 usable labeled sessions. "
        "Therefore, confidence estimation operates as an experimental conversational feature "
        "baseline and must not be presented as a validated psychometric measurement."
    )

    return findings


def train_pilot_confidence_model(
    pilot_path: Path = ROOT_DIR / "data" / "raw" / "edulens_expanded_pilot_020_v2.json",
    output_dir: Path = ROOT_DIR / "models" / "confidence",
) -> Optional[Dict[str, Any]]:
    """
    Trains an exploratory classifier on the pilot annotations if available,
    for interface validation and baseline benchmarking.
    """
    if not pilot_path.exists():
        return None

    with open(pilot_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter for sessions with explicit confidence labels (excluding 'unknown')
    labeled_sessions = [d for d in data if d.get("student_confidence") in CONFIDENCE_LEVELS]
    if len(labeled_sessions) < 6:
        print("Insufficient labeled sessions in pilot to train supervised model.")
        return None

    from ml_prototype.data_loader import extract_student_messages

    rows = []
    for d in labeled_sessions:
        msgs = extract_student_messages(d)
        sig = extract_confidence_signals(msgs)
        sig["target"] = d["student_confidence"]
        sig["student_id"] = d.get("student_id", "UNKNOWN")
        rows.append(sig)

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ["target", "student_id"]]
    X = df[feature_cols]
    y = df["target"]

    clf = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", random_state=42, max_iter=500)),
    ])
    clf.fit(X, y)
    y_pred = clf.predict(X)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "model.joblib"
    joblib.dump(clf, artifact_path)

    metrics = {
        "status": "exploratory_pilot_trained",
        "training_samples": len(df),
        "classes": list(clf.classes_),
        "train_accuracy": float(accuracy_score(y, y_pred)),
        "train_macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "artifact_path": str(artifact_path),
        "note": "Trained on small 10-session pilot subset. Used for interface verification only.",
    }

    report_path = ROOT_DIR / "reports" / "confidence_pilot_evaluation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    print("Auditing confidence datasets...")
    audit = audit_available_confidence_datasets()
    print(json.dumps(audit, indent=2))

    print("\nTraining exploratory pilot model...")
    res = train_pilot_confidence_model()
    print("Pilot train result:", res)

    print("\nTesting inference on sample utterances:")
    estimator = ExpressedConfidenceEstimator()
    sample_hesitant = [
        "uh... i think recursion might be calling itself?",
        "not sure though... is that right? i am a bit lost...",
        "can you give me a hint? idk how to write the base case...",
    ]
    sample_assertive = [
        "definitely! the base case stops the call stack from overflowing.",
        "obviously binary search divides the range in half at every step.",
        "so basically T(n) = T(n/2) + O(1), which gives O(log n) time complexity.",
    ]
    print("Hesitant:", estimator.predict(sample_hesitant))
    print("Assertive:", estimator.predict(sample_assertive))
