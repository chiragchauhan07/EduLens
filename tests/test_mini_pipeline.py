"""
test_mini_pipeline.py — Comprehensive tests for EduLens ML & NLP pipeline.

Verifies:
- JSON conversation parsing across diverse schemas
- Preprocessing and feature extraction
- Understanding classifier inference and probabilities
- Expressed linguistic confidence estimation
- Edge case handling (empty, invalid JSON, missing student, assistant only)
- The 8 practical showcase test conversations
"""

import json
import pytest
from pathlib import Path

from backend.services.conversation_parser import parse_conversation_json, ParsedConversation
from backend.services.preprocessing import prepare_inference_frame, clean_text
from backend.services.feature_extraction import (
    extract_conversational_stats,
    extract_linguistic_confidence_signals,
    estimate_linguistic_confidence,
)
from backend.services.prediction import (
    ModelRegistry,
    analyze_conversation_session,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PARSER & SCHEMA TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_standard_messages_schema():
    payload = {
        "topic": "Binary Search",
        "messages": [
            {"role": "assistant", "content": "How does binary search work?"},
            {"role": "student", "content": "It divides the sorted array in half repeatedly."},
        ],
    }
    parsed = parse_conversation_json(payload)
    assert parsed.topic == "Binary Search"
    assert len(parsed.student_messages) == 1
    assert len(parsed.assistant_messages) == 1
    assert "divides" in parsed.student_messages[0]


def test_parse_turns_schema():
    payload = {
        "concept": "Newton's Laws",
        "turns": [
            {
                "student_message": "Force equals mass times acceleration, right?",
                "tutor_message": "Yes, F = ma.",
            }
        ],
    }
    parsed = parse_conversation_json(payload)
    assert parsed.topic == "Newton's Laws"
    assert len(parsed.student_messages) == 1
    assert len(parsed.assistant_messages) == 1


def test_parse_separate_student_tutor_turns():
    payload = {
        "title": "Photosynthesis",
        "student_turns": [{"text": "Plants absorb light energy using chlorophyll."}],
        "tutor_turns": [{"text": "Correct, in the thylakoid membranes."}],
    }
    parsed = parse_conversation_json(payload)
    assert parsed.topic == "Photosynthesis"
    assert len(parsed.student_messages) == 1
    assert len(parsed.assistant_messages) == 1


def test_parse_raw_json_bytes_and_override_topic():
    raw_bytes = json.dumps({
        "messages": [
            {"role": "user", "text": "Can you explain recursion?"},
            {"role": "ai", "text": "Recursion is when a function calls itself."},
            {"role": "user", "text": "Got it, and it needs a base case to terminate."},
        ]
    }).encode("utf-8")

    parsed = parse_conversation_json(raw_bytes, override_topic="Recursion Concepts")
    assert parsed.topic == "Recursion Concepts"
    assert len(parsed.student_messages) == 2
    assert len(parsed.assistant_messages) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. VALIDATION & ERROR HANDLING TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_json_string_fails():
    with pytest.raises(ValueError, match="empty"):
        parse_conversation_json("")


def test_invalid_json_syntax_fails():
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_conversation_json(b"{invalid json content")


def test_missing_student_turns_fails():
    payload = {"topic": "Math", "messages": []}
    with pytest.raises(ValueError, match="No student messages"):
        parse_conversation_json(payload)


def test_assistant_only_conversation_fails():
    payload = {
        "topic": "History",
        "messages": [
            {"role": "assistant", "content": "Welcome to history class."},
            {"role": "tutor", "content": "Let's begin our discussion today."},
        ],
    }
    with pytest.raises(ValueError, match="only assistant/tutor"):
        parse_conversation_json(payload)


def test_oversized_payload_fails():
    huge_bytes = b"{" + b"a" * (6 * 1024 * 1024) + b"}"
    with pytest.raises(ValueError, match="exceeds the maximum"):
        parse_conversation_json(huge_bytes)


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE EXTRACTION & NLP PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def test_preprocessing_frame_format():
    msgs = ["first student message", "second student message"]
    df, full_text = prepare_inference_frame(msgs)
    assert "full_text" in df.columns
    assert "student_messages" in df.columns
    assert df.iloc[0]["full_text"] == "first student message second student message"
    assert len(df.iloc[0]["student_messages"]) == 2


def test_conversational_stats_extraction():
    msgs = [
        "What does this mean? Can you explain again?",
        "I am so confused, I don't understand how def func(): return True works",
    ]
    stats = extract_conversational_stats(msgs)
    assert stats["num_turns"] == 2.0
    assert stats["question_count"] == 2.0
    assert stats["confusion_markers"] >= 1.0
    assert stats["code_indicators"] >= 1.0


def test_linguistic_confidence_estimation_logic():
    # Hesitant / uncertain utterances
    hesitant_msgs = [
        "um... I'm not sure, maybe it's O(n)?",
        "i guess so... but i don't know, is that right?",
    ]
    label_low, score_low, expl_low, sigs_low = estimate_linguistic_confidence(hesitant_msgs)
    assert label_low in ["Low Confidence", "Moderate Confidence"]
    assert sigs_low["uncertainty_count"] >= 1.0
    assert sigs_low["hesitation_count"] >= 1.0

    # Confident / assertive utterances
    confident_msgs = [
        "I am certain that the search space halves at every step.",
        "Therefore, the maximum number of comparisons is log2(N), which proves O(log n).",
    ]
    label_high, score_high, expl_high, sigs_high = estimate_linguistic_confidence(confident_msgs)
    assert label_high == "High Confidence"
    assert score_high >= 70.0
    assert sigs_high["certainty_count"] >= 1.0
    assert sigs_high["independent_explanations"] >= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. END-TO-END SHOWCASE PRACTICAL TEST CASES
# ─────────────────────────────────────────────────────────────────────────────

def test_showcase_case1_strong_understanding():
    """Case 1: Student articulates concepts clearly and shows solid comprehension."""
    payload = {
        "topic": "Binary Search Trees",
        "messages": [
            {"role": "assistant", "content": "How do binary search trees maintain ordered elements?"},
            {"role": "student", "content": "can we review binary search trees?"},
            {"role": "student", "content": "so basically the left child is always smaller and right child is larger?"},
            {"role": "student", "content": "got it, that makes complete sense! Now I see how search runs in O(log n). Thanks!"},
        ],
    }
    parsed = parse_conversation_json(payload)
    result = analyze_conversation_session(parsed)

    assert result["success"] is True
    assert result["topic"] == "Binary Search Trees"
    assert result["understanding"]["label"] == "Understood"
    assert result["understanding"]["probability"] > 0.70
    assert result["conversation_stats"]["student_messages"] == 3
    assert len(result["insights"]) > 0


def test_showcase_case2_partial_understanding():
    """Case 2: Student has emerging comprehension with some hesitation."""
    payload = {
        "topic": "Recursion",
        "messages": [
            {"role": "assistant", "content": "Can you trace the call stack for factorial(3)?"},
            {"role": "student", "content": "I understand that factorial(3) calls factorial(2)."},
            {"role": "assistant", "content": "Right, and what happens next?"},
            {"role": "student", "content": "Wait what about the base case? I'm not totally sure when it unwinds."},
            {"role": "assistant", "content": "When n reaches 1, it returns 1 and returns up."},
            {"role": "student", "content": "ah okay, so basically each frame returns its result to the caller."},
        ],
    }
    parsed = parse_conversation_json(payload)
    result = analyze_conversation_session(parsed)

    assert result["success"] is True
    assert result["understanding"]["label"] in ["Partially Understood", "Understood"]
    assert "class_probabilities" in result["understanding"]
    assert 0.0 <= result["understanding"]["probability"] <= 1.0


def test_showcase_case3_confusion():
    """Case 3: Student expresses repeated confusion, lost, and difficulty."""
    payload = {
        "topic": "Pointers and Memory",
        "messages": [
            {"role": "student", "content": "i don't understand pointers at all, it's so confusing"},
            {"role": "student", "content": "what is the difference between * and &? I'm completely lost"},
            {"role": "student", "content": "why does it give a segmentation fault? I don't get this"},
        ],
    }
    parsed = parse_conversation_json(payload)
    result = analyze_conversation_session(parsed)

    assert result["success"] is True
    assert result["understanding"]["label"] == "Confused"
    assert result["understanding"]["probability"] > 0.80
    assert result["conversation_stats"]["confusion_signals"] >= 2


def test_showcase_case4_low_expressed_confidence():
    """Case 4: Student expresses heavy doubt, hedging, and low confidence."""
    payload = {
        "topic": "Chemical Bonding",
        "messages": [
            {"role": "student", "content": "um... maybe it forms a covalent bond? I'm not sure at all."},
            {"role": "student", "content": "idk... i guess the electrons are shared? is this correct?"},
            {"role": "student", "content": "i'm completely blank... just give me a hint please."},
        ],
    }
    parsed = parse_conversation_json(payload)
    result = analyze_conversation_session(parsed)

    assert result["success"] is True
    assert result["expressed_confidence"]["label"] == "Low Confidence"
    assert result["expressed_confidence"]["score"] < 40.0
    assert result["expressed_confidence"]["method"] == "linguistic_estimation"
    assert result["expressed_confidence"]["signals"]["uncertainty_count"] >= 2


def test_showcase_case5_high_expressed_confidence():
    """Case 5: Student expresses certainty and assertive arguments."""
    payload = {
        "topic": "Kinematics",
        "messages": [
            {"role": "student", "content": "I am certain that the acceleration is constant due to gravity."},
            {"role": "student", "content": "Therefore, we definitely use v = u + at, which clearly gives 19.6 m/s."},
            {"role": "student", "content": "The result is confirmed without doubt because air resistance is neglected."},
        ],
    }
    parsed = parse_conversation_json(payload)
    result = analyze_conversation_session(parsed)

    assert result["success"] is True
    assert result["expressed_confidence"]["label"] == "High Confidence"
    assert result["expressed_confidence"]["score"] >= 70.0
    assert result["expressed_confidence"]["signals"]["certainty_count"] >= 2

