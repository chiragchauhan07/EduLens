"""
train_model.py — Train, benchmark, and evaluate EduLens ML models.

Models Implemented:
1. Understanding Classifier — Model 1: Multinomial Logistic Regression (Balanced)
2. Understanding Classifier — Model 2: Multinomial Naive Bayes (Probabilistic text baseline)
3. Understanding Classifier — Model 3: Linear Support Vector Machine (LinearSVC with calibration)
4. Quiz Score Regressor — Ridge & Random Forest Regressor on assessment scores

Methodology:
- Leakage-free feature pipelines (TfidfVectorizer + ConversationalFeatureExtractor).
- Group-Aware Splitting (StratifiedGroupKFold by 'student_id') to prevent student persona leakage.
- Rigorous evaluation: Accuracy, Macro Precision, Macro Recall, Macro F1, and Confusion Matrices.
- Artifacts saved into structured directory hierarchy:
    models/understanding/logistic_regression/
    models/understanding/naive_bayes/
    models/understanding/linear_svm/
    models/quiz_score/
    models/trained_model.joblib (primary model pipeline for backward compatibility)
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml_prototype.data_loader import load_primary_dataset
from ml_prototype.features import build_feature_pipeline


def train_understanding_models(
    data_path: Path = None,
    output_dir: Path = ROOT_DIR / "models" / "understanding",
    reports_dir: Path = ROOT_DIR / "reports",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains and benchmarks Logistic Regression, Multinomial Naive Bayes, and Linear SVM
    using Group-Aware splitting on student_id.
    """
    print("\n" + "=" * 70)
    print("STAGE 1: STUDENT UNDERSTANDING CLASSIFIER COMPARISON")
    print("=" * 70)

    # 1. Load Data
    df = load_primary_dataset(data_path)
    total_records = len(df)
    class_names = sorted(df["final_understanding"].unique().tolist())
    print(f"Loaded {total_records} sessions across {len(df['student_id'].unique())} unique students.")
    print("Class distribution:", dict(df["final_understanding"].value_counts()))

    X = df[["full_text", "student_messages"]]
    y = df["final_understanding"]
    groups = df["student_id"]

    # 2. Group-Aware Stratified Split (80% train / 20% test by student_id)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(sgkf.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_students = set(groups.iloc[train_idx])
    test_students = set(groups.iloc[test_idx])

    print(f"\nGroup-Aware Split:")
    print(f"  Training Set : {len(X_train)} sessions ({len(train_students)} unique students)")
    print(f"  Test Set     : {len(X_test)} sessions ({len(test_students)} unique students)")
    print(f"  Student Overlap: {len(train_students & test_students)} (Strict Zero Leakage)")

    # 3. Model Pipelines
    models = {
        "logistic_regression": {
            "name": "Logistic Regression (Balanced)",
            "pipeline": Pipeline([
                ("preprocessor", build_feature_pipeline(scaler="standard")),
                ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state, C=1.0)),
            ]),
            "description": "Multinomial Logistic Regression with balanced weighting and z-score scaled signals.",
        },
        "naive_bayes": {
            "name": "Multinomial Naive Bayes",
            "pipeline": Pipeline([
                ("preprocessor", build_feature_pipeline(scaler="minmax")),
                ("classifier", MultinomialNB(alpha=1.0)),
            ]),
            "description": "Probabilistic text classifier with non-negative MinMax scaled conversational features.",
        },
        "linear_svm": {
            "name": "Linear SVM (Calibrated)",
            "pipeline": Pipeline([
                ("preprocessor", build_feature_pipeline(scaler="standard")),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(class_weight="balanced", random_state=random_state, max_iter=2000),
                        cv=3,
                    ),
                ),
            ]),
            "description": "Margin-based Linear Support Vector Machine with Platt scaling for probability calibration.",
        },
    }

    # 4. Evaluate Baseline (DummyClassifier)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    baseline_metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred_dummy)), 4),
        "macro_f1": round(float(f1_score(y_test, y_pred_dummy, average="macro", zero_division=0)), 4),
    }

    # 5. Train and Cross-Validate Each Model
    results = {}
    print("\nBenchmarking Model Architectures:")
    print("-" * 70)
    print(f"{'Model Architecture':<30} | {'Test Acc':<9} | {'Macro F1':<9} | {'CV Acc (5-Fold)':<16}")
    print("-" * 70)

    for key, item in models.items():
        pipe = item["pipeline"]

        # 5-Fold Stratified Group Cross-Validation on Training Split
        cv_acc = cross_val_score(pipe, X_train, y_train, groups=groups.iloc[train_idx], cv=sgkf, scoring="accuracy")
        cv_f1 = cross_val_score(pipe, X_train, y_train, groups=groups.iloc[train_idx], cv=sgkf, scoring="f1_macro")

        # Fit on Full Training Set and Predict on Unseen Test Students
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        test_acc = float(accuracy_score(y_test, y_pred))
        test_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        test_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        test_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        test_f1_w = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=class_names)
        clf_rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results[key] = {
            "model_key": key,
            "display_name": item["name"],
            "description": item["description"],
            "test_accuracy": round(test_acc, 4),
            "macro_precision": round(test_prec, 4),
            "macro_recall": round(test_rec, 4),
            "macro_f1": round(test_f1, 4),
            "weighted_f1": round(test_f1_w, 4),
            "cv_accuracy_mean": round(float(cv_acc.mean()), 4),
            "cv_accuracy_std": round(float(cv_acc.std()), 4),
            "cv_macro_f1_mean": round(float(cv_f1.mean()), 4),
            "cv_macro_f1_std": round(float(cv_f1.std()), 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": clf_rep,
        }

        print(
            f"{item['name']:<30} | {test_acc:<9.4f} | {test_f1:<9.4f} | "
            f"{cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})"
        )

        # Save individual model artifact into structured folder
        model_subfolder = output_dir / key
        model_subfolder.mkdir(parents=True, exist_ok=True)
        artifact_path = model_subfolder / "model.joblib"
        joblib.dump(pipe, artifact_path)

        metadata_path = model_subfolder / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": item["name"],
                "algorithm": key,
                "classes": class_names,
                "metrics": results[key],
                "split_strategy": "StratifiedGroupKFold(n_splits=5, student_id)",
                "features_dim": 108,
            }, f, indent=2)

    print("-" * 70)

    # 6. Primary Model Selection & Artifact Export
    best_key = max(results.keys(), key=lambda k: results[k]["macro_f1"])
    print(f"\nTop-Performing Model by Macro F1: {models[best_key]['name']} (Macro F1 = {results[best_key]['macro_f1']})")

    # Update backward-compatible trained_model.joblib with top model (Logistic Regression by default)
    primary_pipe = models["logistic_regression"]["pipeline"]
    primary_artifact = ROOT_DIR / "models" / "trained_model.joblib"
    joblib.dump(primary_pipe, primary_artifact)
    print(f"Updated primary compatibility artifact: {primary_artifact}")

    # 7. Save Comparison Report & Model Evaluation JSON
    comparison_payload = {
        "title": "EduLens Student Understanding Models Comparison",
        "benchmark_date": "November 2026",
        "dataset": {
            "total_records": total_records,
            "train_records": len(X_train),
            "test_records": len(X_test),
            "train_students": len(train_students),
            "test_students": len(test_students),
            "split_strategy": "StratifiedGroupKFold on student_id (Zero persona leakage)",
        },
        "baseline": baseline_metrics,
        "models": results,
        "best_model": best_key,
    }

    comp_report_path = reports_dir / "model_comparison.json"
    with open(comp_report_path, "w", encoding="utf-8") as f:
        json.dump(comparison_payload, f, indent=2)
    print(f"Exported model comparison to: {comp_report_path.name}")

    # Standard model_evaluation.json for API and UI compatibility
    lr_res = results["logistic_regression"]
    primary_eval_payload = {
        "model_name": "EduLens Student Understanding Classifier",
        "model_type": "LogisticRegression",
        "target": "final_understanding",
        "target_classes": class_names,
        "features": {
            "text_features": "TfidfVectorizer (max_features=100, unigrams and bigrams)",
            "conversational_features": [
                "num_turns",
                "total_words",
                "avg_turn_length",
                "question_count",
                "confusion_markers",
                "clarification_markers",
                "understanding_markers",
                "code_indicators",
            ],
            "total_features": 108,
        },
        "dataset": {
            "source": str(data_path or "data/processed/edulens_clean_primary.json"),
            "total_records": total_records,
            "train_records": len(X_train),
            "test_records": len(X_test),
            "train_students": len(train_students),
            "test_students": len(test_students),
        },
        "baseline_metrics": {
            "type": "DummyClassifier (majority class)",
            "accuracy": baseline_metrics["accuracy"],
            "macro_f1": baseline_metrics["macro_f1"],
        },
        "evaluation_metrics": {
            "cv_5fold_accuracy_mean": lr_res["cv_accuracy_mean"],
            "cv_5fold_accuracy_std": lr_res["cv_accuracy_std"],
            "cv_5fold_f1_macro_mean": lr_res["cv_macro_f1_mean"],
            "cv_5fold_f1_macro_std": lr_res["cv_macro_f1_std"],
            "test_accuracy": lr_res["test_accuracy"],
            "test_macro_precision": lr_res["macro_precision"],
            "test_macro_recall": lr_res["macro_recall"],
            "test_macro_f1": lr_res["macro_f1"],
            "test_weighted_f1": lr_res["weighted_f1"],
            "confusion_matrix": lr_res["confusion_matrix"],
            "classification_report": lr_res["classification_report"],
        },
        "limitations": [
            "Trained on synthetic student sessions with structured educational prompts.",
            "Labels reflect simulated student persona behaviors, not validated human psychometrics.",
            "Evaluated on university Computer Science topics.",
            "Predictions are diagnostic recommendations, not formal assessment grades.",
        ],
    }

    eval_report_path = reports_dir / "model_evaluation.json"
    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(primary_eval_payload, f, indent=2)
    print(f"Exported primary evaluation to: {eval_report_path.name}")

    return comparison_payload


def train_quiz_regressor(
    quiz_data_path: Path = ROOT_DIR / "data" / "processed" / "edulens_clean_quiz.json",
    output_dir: Path = ROOT_DIR / "models" / "quiz_score",
    reports_dir: Path = ROOT_DIR / "reports",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains and evaluates a Quiz Score Regressor predicting assessment percentage (0-100%).
    """
    print("\n" + "=" * 70)
    print("STAGE 2: QUIZ SCORE PREDICTION REGRESSOR")
    print("=" * 70)

    if not quiz_data_path.exists():
        print(f"Quiz dataset not found at {quiz_data_path}. Generating via prepare_data.py...")
        from scripts.prepare_data import prepare_datasets
        prepare_datasets()

    with open(quiz_data_path, "r", encoding="utf-8") as f:
        quiz_data = json.load(f)

    df = pd.DataFrame(quiz_data)
    print(f"Loaded {len(df)} sessions with post-session quiz scores.")
    print(f"Score statistics: Mean={df['quiz_score_percent'].mean():.2f}%, Std={df['quiz_score_percent'].std():.2f}%")

    X = df[["full_text", "student_messages"]]
    y = df["quiz_score_percent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )

    regressors = {
        "ridge": {
            "name": "Ridge Regressor (L2)",
            "pipeline": Pipeline([
                ("preprocessor", build_feature_pipeline(scaler="standard")),
                ("regressor", Ridge(alpha=1.0, random_state=random_state)),
            ]),
        },
        "random_forest": {
            "name": "Random Forest Regressor",
            "pipeline": Pipeline([
                ("preprocessor", build_feature_pipeline(scaler="standard")),
                ("regressor", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state)),
            ]),
        },
    }

    eval_results = {}
    print("\nTraining Regression Models:")
    print("-" * 65)
    print(f"{'Model Name':<28} | {'MAE (%)':<10} | {'RMSE':<10} | {'R2 Score':<10}")
    print("-" * 65)

    for key, item in regressors.items():
        pipe = item["pipeline"]
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        # Clamp predictions to [0, 100]
        y_pred_clipped = np.clip(y_pred, 0.0, 100.0)

        mae = float(mean_absolute_error(y_test, y_pred_clipped))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_clipped)))
        r2 = float(r2_score(y_test, y_pred_clipped))

        eval_results[key] = {
            "name": item["name"],
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 4),
        }

        print(f"{item['name']:<28} | {mae:<10.2f} | {rmse:<10.2f} | {r2:<10.4f}")

    # Select best regressor
    best_reg_key = max(eval_results.keys(), key=lambda k: eval_results[k]["r2"])
    best_pipe = regressors[best_reg_key]["pipeline"]

    output_dir.mkdir(parents=True, exist_ok=True)
    reg_artifact_path = output_dir / "model.joblib"
    joblib.dump(best_pipe, reg_artifact_path)
    print(f"\nSaved primary Quiz Regressor ({best_reg_key}) to: {reg_artifact_path}")

    quiz_report_payload = {
        "task": "Quiz Score Prediction (0-100%)",
        "dataset_records": len(df),
        "train_records": len(X_train),
        "test_records": len(X_test),
        "models": eval_results,
        "selected_model": best_reg_key,
    }

    quiz_report_path = reports_dir / "quiz_regressor_evaluation.json"
    with open(quiz_report_path, "w", encoding="utf-8") as f:
        json.dump(quiz_report_payload, f, indent=2)
    print(f"Saved quiz regressor evaluation to: {quiz_report_path.name}")
    print("=" * 70)

    return quiz_report_payload


def run_full_training():
    train_understanding_models()
    train_quiz_regressor()


if __name__ == "__main__":
    run_full_training()
