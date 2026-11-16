"""
feature_extraction.py — Pedagogical and Conversational Feature Extraction for EduLens Mini.

Extracts:
1. 8 Dense Conversational Signals for ML model features
2. Linguistic Confidence Signals for transparent, rule-calibrated student confidence estimation
"""

import re
from typing import Any, Dict, List, Tuple
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1. PEDAGOGICAL PATTERNS (Used in ML model feature space)
# ─────────────────────────────────────────────────────────────────────────────
CONFUSION_PATTERN = re.compile(
    r"\b(confus\w*|lost|stuck|don'?t\s+understand|don'?t\s+get|not\s+getting|didn'?t\s+get|"
    r"doubt|hard|unclear|struggl\w*|what\s+does\s+that\s+mean|huh|i'?m\s+blank|wait\s+what)\b",
    re.IGNORECASE,
)

CLARIFICATION_PATTERN = re.compile(
    r"\b(explain\s+again|another\s+example|give\s+an\s+example|can\s+you\s+show|"
    r"clarif\w*|repeat|difference\s+between|why\s+does|how\s+come|step\s+by\s+step)\b",
    re.IGNORECASE,
)

UNDERSTANDING_PATTERN = re.compile(
    r"\b(got\s+it|makes?\s+sense|understood|now\s+i\s+get|i\s+see|clear\s+now|"
    r"so\s+basically|that\s+means|ah\s+okay|oh\s+okay|thank\s+you|thanks|perfect)\b",
    re.IGNORECASE,
)

CODE_SYNTAX_PATTERN = re.compile(
    r"(`{1,3}|[{};]|\b(def|class|return|int|float|void|print|import|for|while)\b|==|!=|->)",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. LINGUISTIC CONFIDENCE PATTERNS (Linguistic Evidence)
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


def extract_conversational_stats(messages: List[str]) -> Dict[str, float]:
    """Computes summary conversational metrics from student utterances."""
    if not messages:
        return {
            "num_turns": 0.0,
            "total_words": 0.0,
            "avg_turn_length": 0.0,
            "question_count": 0.0,
            "confusion_markers": 0.0,
            "clarification_markers": 0.0,
            "understanding_markers": 0.0,
            "code_indicators": 0.0,
        }

    full_text = " ".join(messages)
    words = full_text.split()
    total_words = len(words)
    num_turns = len(messages)
    avg_turn_length = total_words / max(1, num_turns)

    return {
        "num_turns": float(num_turns),
        "total_words": float(total_words),
        "avg_turn_length": float(round(avg_turn_length, 2)),
        "question_count": float(full_text.count("?")),
        "confusion_markers": float(len(CONFUSION_PATTERN.findall(full_text))),
        "clarification_markers": float(len(CLARIFICATION_PATTERN.findall(full_text))),
        "understanding_markers": float(len(UNDERSTANDING_PATTERN.findall(full_text))),
        "code_indicators": float(len(CODE_SYNTAX_PATTERN.findall(full_text))),
    }


def extract_linguistic_confidence_signals(student_messages: List[str]) -> Dict[str, float]:
    """Extracts explicit linguistic markers associated with student expressed certainty and doubt."""
    if not student_messages:
        return {
            "uncertainty_count": 0.0,
            "certainty_count": 0.0,
            "hesitation_count": 0.0,
            "confirmation_requests": 0.0,
            "self_corrections": 0.0,
            "independent_explanations": 0.0,
            "hint_dependencies": 0.0,
            "assertion_hedge_ratio": 0.0,
        }

    full_text = " ".join(student_messages)
    unc = float(len(RE_UNCERTAINTY.findall(full_text)))
    cert = float(len(RE_CERTAINTY.findall(full_text)))
    hes = float(len(RE_HESITATION.findall(full_text)))
    conf_req = float(len(RE_CONFIRMATION_REQUEST.findall(full_text)))
    self_corr = float(len(RE_SELF_CORRECTION.findall(full_text)))
    indep = float(len(RE_INDEPENDENT_EXPLANATION.findall(full_text)))
    hint_dep = float(len(RE_HINT_DEPENDENCE.findall(full_text)))

    hedge_sum = unc + hes + (0.5 * conf_req)
    assert_sum = cert + indep + (0.5 * self_corr)
    ratio = float((assert_sum + 1.0) / (hedge_sum + 1.0))

    return {
        "uncertainty_count": unc,
        "certainty_count": cert,
        "hesitation_count": hes,
        "confirmation_requests": conf_req,
        "self_corrections": self_corr,
        "independent_explanations": indep,
        "hint_dependencies": hint_dep,
        "assertion_hedge_ratio": round(ratio, 2),
    }


def estimate_linguistic_confidence(student_messages: List[str]) -> Tuple[str, float, str, Dict[str, float]]:
    """
    Estimates student expressed confidence from linguistic evidence.

    Returns:
        Tuple of (label: 'High Confidence'|'Moderate Confidence'|'Low Confidence',
                  score: float 0-100,
                  explanation: str,
                  signals: dict)
    """
    signals = extract_linguistic_confidence_signals(student_messages)
    if not student_messages:
        return "Low Confidence", 0.0, "No student utterances to analyze.", signals

    # Baseline neutral score
    score = 55.0

    # Boosters: assertions, independent explanations, self-corrections (metacognition)
    score += signals["certainty_count"] * 9.0
    score += signals["independent_explanations"] * 7.0
    score += signals["self_corrections"] * 5.0

    # Dampeners: uncertainty markers, hesitation, confirmation checks, hint reliance
    score -= signals["uncertainty_count"] * 8.0
    score -= signals["hesitation_count"] * 5.0
    score -= signals["confirmation_requests"] * 4.0
    score -= signals["hint_dependencies"] * 9.0

    # Words per turn bonus/penalty
    words = sum(len(m.split()) for m in student_messages)
    avg_len = words / max(1, len(student_messages))
    if avg_len < 6.0:
        score -= 5.0
    elif avg_len > 18.0:
        score += 4.0

    clamped_score = round(float(np.clip(score, 5.0, 95.0)), 1)

    if clamped_score < 40.0:
        label = "Low Confidence"
        explanation = (
            f"Expressed confidence is low ({clamped_score}%) with {int(signals['uncertainty_count'])} "
            f"uncertainty expressions and {int(signals['hesitation_count'])} hesitation markers."
        )
    elif clamped_score <= 68.0:
        label = "Moderate Confidence"
        explanation = (
            f"Expressed confidence is moderate ({clamped_score}%) reflecting balanced inquiry "
            f"with some tentative phrasing or confirmation questions."
        )
    else:
        label = "High Confidence"
        explanation = (
            f"Expressed confidence is high ({clamped_score}%) with assertive conceptual claims "
            f"and minimal hesitation."
        )

    return label, clamped_score, explanation, signals
