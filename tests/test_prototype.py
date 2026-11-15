"""
test_prototype.py — Comprehensive automated test suite for EduLens ML Prototype.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

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
    df = load_primary_dataset()
    assert len(df) == 430
    assert "final_understanding" in df.columns
    assert set(df["final_understanding"].unique()) == {"understood", "partial", "confused"}
    assert "student_id" in df.columns
    assert len(df["student_id"].unique()) == 30


def test_feature_pipeline_scalers():
    df = load_primary_dataset()
    sample_df = df.head(10)

    # Test standard scaler
    pipe_std = build_feature_pipeline(scaler="standard", max_tfidf_features=50)
    X_std = pipe_std.fit_transform(sample_df)
    assert X_std.shape == (10, 58)  # 50 tfidf + 8 stats

    # Test minmax scaler (ensuring all features >= 0 for Naive Bayes)
    pipe_mm = build_feature_pipeline(scaler="minmax", max_tfidf_features=50)
    X_mm = pipe_mm.fit_transform(sample_df)
    assert X_mm.shape == (10, 58)
    assert float(X_mm.min()) >= 0.0


def test_predictor_primary_inference():
    predictor = EduLensPredictor()
    sample = [
        "I really don't understand how this recursive function unwinds.",
        "I am completely stuck and confused. What does base case mean?",
    ]
    result = predictor.predict(sample)
    assert "predicted_class" in result
    assert result["predicted_class"] in {"understood", "partial", "confused"}
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "probabilities" in result
    assert "conversational_signals" in result
    assert result["conversational_signals"]["num_turns"] == 2.0
    assert "predicted_quiz_score" in result


def test_predictor_all_algorithms():
    for algo in ["logistic_regression", "naive_bayes", "linear_svm"]:
        pred = EduLensPredictor(algorithm=algo)
        sample = {
            "turns": [
                {"student_message": "can you explain merge sort?"},
                {"student_message": "got it, dividing takes log n and merging takes O(n). Makes sense now!"},
            ]
        }
        res = pred.predict(sample)
        assert res["predicted_class"] in {"understood", "partial", "confused"}
        assert len(res["probabilities"]) == 3
        assert sum(res["probabilities"].values()) == pytest.approx(1.0, abs=0.05)


def test_quiz_score_regressor():
    predictor = EduLensPredictor(load_quiz_regressor=True)
    sample = [
        "got it! that makes complete sense! Master theorem case 2 applies directly.",
        "thank you so much, now I understand perfectly!",
    ]
    res = predictor.predict(sample)
    assert res["predicted_quiz_score"] is not None
    assert 0.0 <= res["predicted_quiz_score"] <= 100.0


def test_predictor_invalid_inputs():
    predictor = EduLensPredictor()
    # Empty string
    res_empty = predictor.predict("")
    assert "error" in res_empty
    # Empty list
    res_list = predictor.predict([])
    assert "error" in res_list


def test_zero_student_leakage_split():
    df = load_primary_dataset()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    X = df[["full_text", "student_messages"]]
    y = df["final_understanding"]
    groups = df["student_id"]

    for train_idx, test_idx in sgkf.split(X, y, groups):
        train_students = set(groups.iloc[train_idx])
        test_students = set(groups.iloc[test_idx])
        assert len(train_students & test_students) == 0, "Student persona leakage detected across splits!"
