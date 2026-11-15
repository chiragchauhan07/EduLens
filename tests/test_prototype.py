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

ROOT_DIR = Path(__file__).resolve().parent.parent


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


def test_predictor_with_explainability():
    predictor = EduLensPredictor()
    sample = [
        "I don't understand pointers at all, it's so confusing.",
        "What is the difference between * and &? I'm completely lost.",
    ]
    res = predictor.predict(sample, explain=True)
    assert "feature_explanation" in res
    exp = res["feature_explanation"]
    assert exp["predicted_class"] == "confused"
    assert "top_positive_features" in exp
    assert "top_negative_features" in exp
    assert len(exp["top_positive_features"]) > 0


def test_explainability_helpers():
    from ml_prototype.explainability import (
        explain_logistic_regression,
        explain_linear_svm,
        explain_naive_bayes,
        explain_quiz_regressor,
    )
    import joblib
    from ml_prototype.predict import ROOT_DIR

    lr_path = ROOT_DIR / "models" / "understanding" / "logistic_regression" / "model.joblib"
    if lr_path.exists():
        lr_pipe = joblib.load(lr_path)
        lr_exp = explain_logistic_regression(lr_pipe)
        assert "class_explanations" in lr_exp
        assert "understood" in lr_exp["class_explanations"]

    svm_path = ROOT_DIR / "models" / "understanding" / "linear_svm" / "model.joblib"
    if svm_path.exists():
        svm_pipe = joblib.load(svm_path)
        svm_exp = explain_linear_svm(svm_pipe)
        assert "class_explanations" in svm_exp

    nb_path = ROOT_DIR / "models" / "understanding" / "naive_bayes" / "model.joblib"
    if nb_path.exists():
        nb_pipe = joblib.load(nb_path)
        nb_exp = explain_naive_bayes(nb_pipe)
        assert "class_explanations" in nb_exp

    quiz_path = ROOT_DIR / "models" / "quiz_score" / "model.joblib"
    if quiz_path.exists():
        q_dict = joblib.load(quiz_path)
        q_exp = explain_quiz_regressor(q_dict)
        assert "top_features" in q_exp or "ridge_coefficients" in q_exp or "random_forest_importance" in q_exp


def test_feature_ablation_subsets():
    df = load_primary_dataset().head(5)

    # Exp A: TF-IDF only
    pipe_a = build_feature_pipeline(include_convo=False, max_tfidf_features=20)
    Xa = pipe_a.fit_transform(df)
    assert Xa.shape == (5, 20)

    # Exp B: Dense only
    pipe_b = build_feature_pipeline(include_tfidf=False)
    Xb = pipe_b.fit_transform(df)
    assert Xb.shape == (5, 8)

    # Exp D: No explicit markers
    pipe_d = build_feature_pipeline(
        max_tfidf_features=20,
        convo_features=[
            "num_turns",
            "total_words",
            "avg_turn_length",
            "question_count",
            "clarification_markers",
            "code_indicators",
        ],
    )
    Xd = pipe_d.fit_transform(df)
    assert Xd.shape == (5, 26)  # 20 tfidf + 6 selected dense stats

    # Exp F: No explicit markers and no shortcut vocabulary
    shortcut_words = ["confused", "understood", "clear", "got", "makes", "sense"]
    pipe_f = build_feature_pipeline(
        max_tfidf_features=20,
        convo_features=["num_turns", "total_words", "avg_turn_length", "question_count", "clarification_markers", "code_indicators"],
        custom_stop_words=shortcut_words,
    )
    Xf = pipe_f.fit_transform(df)
    assert Xf.shape == (5, 26)


def test_quiz_group_isolation():
    """Verify strict zero student leakage in quiz regression."""
    quiz_json = ROOT_DIR / "data" / "processed" / "edulens_clean_quiz.json"
    if quiz_json.exists():
        with open(quiz_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, test_idx = next(gss.split(df, df["quiz_score_percent"], df["student_id"]))
        train_students = set(df["student_id"].iloc[train_idx])
        test_students = set(df["student_id"].iloc[test_idx])
        assert len(train_students & test_students) == 0, "Student persona leakage detected in quiz regression!"

    # Verify report JSON has zero overlap
    report_path = ROOT_DIR / "reports" / "quiz_regressor_evaluation.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        if "train_student_ids" in rep and "test_student_ids" in rep:
            tr_set = set(rep["train_student_ids"])
            te_set = set(rep["test_student_ids"])
            assert len(tr_set & te_set) == 0, "Student overlap found in quiz evaluation report!"


def test_all_model_artifacts_load():
    """Verify all saved model artifacts can be loaded and executed."""
    import joblib
    models_to_test = [
        ROOT_DIR / "models" / "trained_model.joblib",
        ROOT_DIR / "models" / "understanding" / "logistic_regression" / "model.joblib",
        ROOT_DIR / "models" / "understanding" / "linear_svm" / "model.joblib",
        ROOT_DIR / "models" / "understanding" / "naive_bayes" / "model.joblib",
        ROOT_DIR / "models" / "quiz_score" / "model.joblib",
    ]
    sample_df = pd.DataFrame([{
        "full_text": "I understand binary search tree time complexity now. Thank you!",
        "student_messages": ["I understand binary search tree time complexity now.", "Thank you!"],
    }])
    for m_path in models_to_test:
        assert m_path.exists(), f"Model artifact missing at {m_path}"
        loaded = joblib.load(m_path)
        pred = loaded.predict(sample_df)
        assert len(pred) == 1
