"""
scripts/run_feature_ablation.py — Systematic Feature Ablation Experiments for EduLens.

Evaluates 5 Feature Configurations across understanding classification:
- Experiment A: TF-IDF only (100 lexical features).
- Experiment B: Eight conversational features only (8 dense signals).
- Experiment C: All 108 combined features (Full feature set).
- Experiment D: All features except explicit understanding and confusion markers (106 features).
- Experiment E: All features except the eight dense conversational signals (Ablates conversational metrics).

Evaluates on the exact same Group-Aware test split (StratifiedGroupKFold on student_id)
with strict zero student leakage.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml_prototype.data_loader import load_primary_dataset
from ml_prototype.features import build_feature_pipeline


def run_ablation_experiments(random_state: int = 42) -> Dict[str, Any]:
    print("=" * 75)
    print("EDULENS FEATURE ABLATION EXPERIMENTS")
    print("=" * 75)

    df = load_primary_dataset()
    X = df[["full_text", "student_messages"]]
    y = df["final_understanding"]
    groups = df["student_id"]
    class_names = sorted(y.unique().tolist())

    # Same group-aware split as primary benchmark
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(sgkf.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_students = set(groups.iloc[train_idx])
    test_students = set(groups.iloc[test_idx])
    print(f"Data Partitioning (Strict Zero Student Leakage):")
    print(f"  Training: {len(X_train)} sessions ({len(train_students)} students)")
    print(f"  Testing : {len(X_test)} sessions ({len(test_students)} students)")
    print(f"  Student Overlap: {len(train_students & test_students)}")

    # Define the 5 Ablation Configurations
    experiments = {
        "Exp_A_tfidf_only": {
            "name": "Experiment A: TF-IDF Only",
            "description": "Lexical n-grams only (100 features); all conversational and structural metrics removed.",
            "pipeline_args": {"include_tfidf": True, "include_convo": False, "scaler": "standard"},
            "expected_dim": 100,
        },
        "Exp_B_convo_only": {
            "name": "Experiment B: Eight Conversational Signals Only",
            "description": "Dense dialogue structural and pedagogical metrics only (8 features); all vocabulary removed.",
            "pipeline_args": {"include_tfidf": False, "include_convo": True, "scaler": "standard"},
            "expected_dim": 8,
        },
        "Exp_C_all_108": {
            "name": "Experiment C: All 108 Combined Features",
            "description": "Full feature union (100 TF-IDF + 8 scaled conversational signals).",
            "pipeline_args": {"include_tfidf": True, "include_convo": True, "scaler": "standard"},
            "expected_dim": 108,
        },
        "Exp_D_no_explicit_markers": {
            "name": "Experiment D: All Features Except Explicit Understanding/Confusion Markers",
            "description": "Excludes 'confusion_markers' and 'understanding_markers' from dense signals (106 features: 100 TF-IDF + 6 structural).",
            "pipeline_args": {
                "include_tfidf": True,
                "include_convo": True,
                "convo_features": ["num_turns", "total_words", "avg_turn_length", "question_count", "clarification_markers", "code_indicators"],
                "scaler": "standard",
            },
            "expected_dim": 106,
        },
        "Exp_E_no_convo_signals": {
            "name": "Experiment E: All Features Except Eight Dense Conversational Signals",
            "description": "Ablates all dense structural conversational signals, relying strictly on vocabulary context.",
            "pipeline_args": {"include_tfidf": True, "include_convo": False, "scaler": "standard"},
            "expected_dim": 100,
        },
        "Exp_F_no_explicit_markers_nor_shortcut_vocab": {
            "name": "Experiment F: No Explicit Markers AND No Shortcut Vocabulary in TF-IDF",
            "description": (
                "Removes explicit dense counters (understanding_markers, confusion_markers) AND filters out all "
                "shortcut pedagogical terms from TF-IDF vocabulary (e.g. 'confused', 'understood', 'clear', 'lost', "
                "'stuck', 'thank', 'perfect', 'sorry', 'doubt', 'unclear', 'ok', 'okay'). Tests pure topical and structural reasoning."
            ),
            "pipeline_args": {
                "include_tfidf": True,
                "include_convo": True,
                "convo_features": ["num_turns", "total_words", "avg_turn_length", "question_count", "clarification_markers", "code_indicators"],
                "custom_stop_words": [
                    "confused", "confusion", "lost", "stuck", "understand", "understood", "understanding",
                    "doubt", "unclear", "struggle", "struggling", "clear", "got", "makes", "sense",
                    "okay", "ok", "thank", "thanks", "perfect", "sorry", "wait", "huh", "explain", "clarify", "repeat",
                ],
                "scaler": "standard",
            },
            "expected_dim": 106,
        },
    }

    ablation_results = {}
    print("\nRunning Ablation Evaluation (Logistic Regression & Linear SVM):")
    print("-" * 75)
    print(f"{'Experiment':<35} | {'Feats':<5} | {'LR Acc':<8} | {'LR F1':<8} | {'SVM Acc':<8} | {'SVM F1':<8}")
    print("-" * 75)

    for exp_key, exp_info in experiments.items():
        preprocessor = build_feature_pipeline(**exp_info["pipeline_args"])

        # Check feature dimension
        preprocessor.fit(X_train)
        X_tr_trans = preprocessor.transform(X_train)
        dim = X_tr_trans.shape[1]

        # 1. Logistic Regression Model
        pipe_lr = Pipeline([
            ("preprocessor", build_feature_pipeline(**exp_info["pipeline_args"])),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state, C=1.0)),
        ])
        pipe_lr.fit(X_train, y_train)
        y_pred_lr = pipe_lr.predict(X_test)

        lr_acc = float(accuracy_score(y_test, y_pred_lr))
        lr_prec = float(precision_score(y_test, y_pred_lr, average="macro", zero_division=0))
        lr_rec = float(recall_score(y_test, y_pred_lr, average="macro", zero_division=0))
        lr_f1 = float(f1_score(y_test, y_pred_lr, average="macro", zero_division=0))
        lr_cm = confusion_matrix(y_test, y_pred_lr, labels=class_names)
        lr_rep = classification_report(y_test, y_pred_lr, output_dict=True, zero_division=0)

        # 2. Linear SVM Model
        pipe_svm = Pipeline([
            ("preprocessor", build_feature_pipeline(**exp_info["pipeline_args"])),
            ("classifier", CalibratedClassifierCV(
                estimator=LinearSVC(class_weight="balanced", random_state=random_state, max_iter=2000),
                cv=3,
            )),
        ])
        pipe_svm.fit(X_train, y_train)
        y_pred_svm = pipe_svm.predict(X_test)

        svm_acc = float(accuracy_score(y_test, y_pred_svm))
        svm_prec = float(precision_score(y_test, y_pred_svm, average="macro", zero_division=0))
        svm_rec = float(recall_score(y_test, y_pred_svm, average="macro", zero_division=0))
        svm_f1 = float(f1_score(y_test, y_pred_svm, average="macro", zero_division=0))
        svm_cm = confusion_matrix(y_test, y_pred_svm, labels=class_names)
        svm_rep = classification_report(y_test, y_pred_svm, output_dict=True, zero_division=0)

        ablation_results[exp_key] = {
            "name": exp_info["name"],
            "description": exp_info["description"],
            "feature_count": dim,
            "logistic_regression": {
                "accuracy": round(lr_acc, 4),
                "macro_precision": round(lr_prec, 4),
                "macro_recall": round(lr_rec, 4),
                "macro_f1": round(lr_f1, 4),
                "confusion_matrix": lr_cm.tolist(),
                "classification_report": lr_rep,
            },
            "linear_svm": {
                "accuracy": round(svm_acc, 4),
                "macro_precision": round(svm_prec, 4),
                "macro_recall": round(svm_rec, 4),
                "macro_f1": round(svm_f1, 4),
                "confusion_matrix": svm_cm.tolist(),
                "classification_report": svm_rep,
            },
        }

        print(
            f"{exp_info['name']:<35} | {dim:<5} | "
            f"{lr_acc:<8.4f} | {lr_f1:<8.4f} | "
            f"{svm_acc:<8.4f} | {svm_f1:<8.4f}"
        )

    print("-" * 75)

    # Save Ablation Report
    report_path = ROOT_DIR / "reports" / "feature_ablation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "title": "EduLens Feature Ablation Report",
            "evaluation_protocol": "StratifiedGroupKFold on student_id (Test N=73, Train N=357, Student Overlap=0)",
            "target_classes": class_names,
            "experiments": ablation_results,
        }, f, indent=2)
    print(f"\nSaved ablation report to: {report_path}")

    return ablation_results


if __name__ == "__main__":
    run_ablation_experiments()

