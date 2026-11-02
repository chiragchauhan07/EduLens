"""
test_prototype.py — Automated test suite for EduLens ML Prototype.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml_prototype.data_loader import extract_student_messages, normalize_session, load_primary_dataset
from ml_prototype.features import (
    extract_conversation_stats,
    ConversationalFeatureExtractor,
    build_feature_pipeline,
)
from ml_prototype.predict import EduLensPredictor


def test_extract_student_messages_turns():
    record = {
        "turns": [
            {"student_message": "What is recursion?"},
            {"student_message": "Can you show an example?"},
        ]
    }
    messages = extract_student_messages(record)
    assert len(messages) == 2
    assert messages[0] == "What is recursion?"
    assert messages[1] == "Can you show an example?"


def test_extract_conversation_stats():
    messages = [
        "I am totally confused and lost about pointers.",
        "Can you explain again with an example?",
        "Oh got it, that makes sense now! int* ptr = &val;",
    ]
    stats = extract_conversation_stats(messages)
    assert stats["num_turns"] == 3.0
    assert stats["total_words"] > 0
    assert stats["question_count"] == 1.0
    assert stats["confusion_markers"] >= 1.0
    assert stats["clarification_markers"] >= 1.0
    assert stats["understanding_markers"] >= 1.0
    assert stats["code_indicators"] >= 1.0


def test_primary_dataset_loading():
    dataset_path = Path("data/raw/edulens_chats_dataset.json")
    if dataset_path.exists():
        df = load_primary_dataset(dataset_path)
        assert len(df) == 430
        assert "final_understanding" in df.columns
        assert set(df["final_understanding"].unique()) == {"understood", "partial", "confused"}


def test_predictor_inference():
    predictor = EduLensPredictor()
    sample = [
        "I really don't understand how this recursive function unwinds.",
        "I am completely stuck and confused. What does base case mean?",
    ]
    result = predictor.predict(sample)
    assert "predicted_class" in result
    assert result["predicted_class"] in {"understood", "partial", "confused"}
    assert "confidence" in result
    assert "probabilities" in result
    assert "conversational_signals" in result
    assert result["conversational_signals"]["num_turns"] == 2.0
