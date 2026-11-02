"""
train_model.py — Train and evaluate the primary EduLens ML model.

Workflow:
1. Load verified dataset (430 sessions from edulens_chats_dataset.json).
2. Separate feature columns X from target y ('final_understanding').
3. Perform a stratified 80/20 train/test split to prevent leakage.
4. Fit the feature extraction pipeline (TF-IDF + conversational metrics) strictly on train data.
5. Train a primary Multinomial Logistic Regression model with balanced class weighting.
6. Train a DummyClassifier baseline (majority class) for honest comparison.
7. Perform 5-fold Stratified Cross-Validation on training data.
8. Evaluate on the un-seen test split: Accuracy, Precision, Recall, F1, and Confusion Matrix.
9. Save the full fitted pipeline to models/trained_model.joblib.
10. Save the evaluation metrics to reports/model_evaluation.json.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml_prototype.data_loader import load_primary_dataset
from ml_prototype.features import build_feature_pipeline


def train_and_evaluate(
    data_path: str = "data/raw/edulens_chats_dataset.json",
    model_output_path: str = "models/trained_model.joblib",
    report_output_path: str = "reports/model_evaluation.json",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes the full training and evaluation workflow.
    """
    print("=" * 70)
    print("EDULENS ML PROTOTYPE — TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print(f"\n[1/6] Loading primary supervised dataset from: {data_path}")
    df = load_primary_dataset(data_path)
    total_records = len(df)
    print(f"      Loaded {total_records} valid sessions.")
    print("      Class distribution in dataset:")
    for cls, count in df["final_understanding"].value_counts().items():
        pct = (count / total_records) * 100
        print(f"        - {cls:<12}: {count:>3} ({pct:.1f}%)")

    # 2. Features and Target
    X = df[["full_text", "student_messages"]]
    y = df["final_understanding"]
    class_names = sorted(y.unique().tolist())

    # 3. Train / Test Split (Stratified)
    print(f"\n[2/6] Splitting data into 80% train / 20% test (stratified, seed={random_state})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    print(f"      Training set size : {len(X_train)} sessions")
    print(f"      Test set size     : {len(X_test)} sessions")

    # 4. Construct Full Pipeline (Preprocessor + Classifier)
    print("\n[3/6] Building Scikit-learn Pipeline (Preprocessing + Logistic Regression)")
    preprocessor = build_feature_pipeline()
    
    # Primary model: Multinomial Logistic Regression with balanced weighting
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        C=1.0,
    )
    
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    # 5. Cross-Validation on Training Data
    print("\n[4/6] Running 5-Fold Stratified Cross-Validation on Training Set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_acc_scores = cross_val_score(model_pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    cv_f1_scores = cross_val_score(model_pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"      CV Accuracy : {cv_acc_scores.mean():.4f} (+/- {cv_acc_scores.std():.4f})")
    print(f"      CV Macro F1 : {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std():.4f})")

    # 6. Fit Baseline (Dummy Classifier)
    print("\n[5/6] Fitting Baseline Model (Majority Class Dummy Classifier)...")
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)
    baseline_acc = float(accuracy_score(y_test, y_pred_baseline))
    baseline_f1 = float(f1_score(y_test, y_pred_baseline, average="macro", zero_division=0))
    print(f"      Baseline Test Accuracy : {baseline_acc:.4f}")
    print(f"      Baseline Test Macro F1 : {baseline_f1:.4f}")

    # 7. Fit Primary Pipeline on Full Training Set and Evaluate on Test Set
    print("\n[6/6] Fitting Primary Model Pipeline on Training Set & Evaluating on Test Set...")
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)

    # Metrics
    test_acc = float(accuracy_score(y_test, y_pred))
    test_prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    test_rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    test_f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    test_f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print("\n" + "-" * 70)
    print(f"TEST SET EVALUATION RESULTS ({len(X_test)} conversations):")
    print("-" * 70)
    print(f"  Model Type          : Multinomial Logistic Regression (Balanced)")
    print(f"  Test Accuracy       : {test_acc:.4f}  (Baseline: {baseline_acc:.4f})")
    print(f"  Macro Precision     : {test_prec_macro:.4f}")
    print(f"  Macro Recall        : {test_rec_macro:.4f}")
    print(f"  Macro F1-Score      : {test_f1_macro:.4f}  (Baseline: {baseline_f1:.4f})")
    print(f"  Weighted F1-Score   : {test_f1_weighted:.4f}")
    print("\n  Per-Class Performance:")
    for cls in class_names:
        stats = clf_report.get(cls, {})
        print(
            f"    Class '{cls:<10}': Precision={stats.get('precision', 0):.3f}, "
            f"Recall={stats.get('recall', 0):.3f}, F1={stats.get('f1-score', 0):.3f}, "
            f"Support={stats.get('support', 0)}"
        )
    print("\n  Confusion Matrix (rows: true, cols: pred):")
    print(f"    Labels: {class_names}")
    for i, row in enumerate(cm):
        print(f"    {class_names[i]:<10}: {row}")

    # 8. Save Artifacts
    # Ensure destination directories exist
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)

    print(f"\nSaving model pipeline to: {model_output_path}")
    joblib.dump(model_pipeline, model_output_path)

    evaluation_payload = {
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
            "source": data_path,
            "total_records": total_records,
            "train_records": len(X_train),
            "test_records": len(X_test),
        },
        "baseline_metrics": {
            "type": "DummyClassifier (majority class)",
            "accuracy": round(baseline_acc, 4),
            "macro_f1": round(baseline_f1, 4),
        },
        "evaluation_metrics": {
            "cv_5fold_accuracy_mean": round(float(cv_acc_scores.mean()), 4),
            "cv_5fold_accuracy_std": round(float(cv_acc_scores.std()), 4),
            "cv_5fold_f1_macro_mean": round(float(cv_f1_scores.mean()), 4),
            "cv_5fold_f1_macro_std": round(float(cv_f1_scores.std()), 4),
            "test_accuracy": round(test_acc, 4),
            "test_macro_precision": round(test_prec_macro, 4),
            "test_macro_recall": round(test_rec_macro, 4),
            "test_macro_f1": round(test_f1_macro, 4),
            "test_weighted_f1": round(test_f1_weighted, 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": clf_report,
        },
        "limitations": [
            "Trained on synthetic Claude-generated student sessions.",
            "Labels reflect synthetic student persona simulation, not validated educational psychometrics.",
            "Domain is focused on university computer science topics.",
            "Predictions should guide instructional interventions, not replace formal assessments.",
        ],
    }

    print(f"Saving evaluation report to: {report_output_path}")
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_payload, f, indent=2)

    print("\nTraining and evaluation successfully completed.")
    print("=" * 70)
    return evaluation_payload


if __name__ == "__main__":
    train_and_evaluate()
