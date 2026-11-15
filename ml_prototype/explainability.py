"""
explainability.py — Practical Model Explainability for EduLens.

Provides deterministic, coefficient-based and feature-importance explanations for:
1. Multinomial Logistic Regression (Class-specific positive and negative coefficients).
2. Linear SVM (Decision hyperplane weights per class).
3. Multinomial Naive Bayes (Log-likelihood ratio and most distinctive terms per class).
4. Quiz Score Regressor (Feature importance & directional weights).
5. Instance-Level Local Explanation (Decomposes any single conversation into driving features).
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml_prototype.data_loader import extract_student_messages
from ml_prototype.features import extract_conversation_stats


def get_pipeline_feature_names(preprocessor) -> List[str]:
    """Extracts human-readable feature names from the ColumnTransformer."""
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "text_tfidf":
            tfidf_feats = [f"tfidf_{w}" for w in trans.get_feature_names_out()]
            feature_names.extend(tfidf_feats)
        elif name == "convo_stats":
            # Extract step from pipeline
            extractor = trans.named_steps.get("extract")
            if extractor and hasattr(extractor, "get_feature_names_out"):
                convo_feats = [f"signal_{f}" for f in extractor.get_feature_names_out()]
            else:
                convo_feats = [
                    "signal_num_turns",
                    "signal_total_words",
                    "signal_avg_turn_length",
                    "signal_question_count",
                    "signal_confusion_markers",
                    "signal_clarification_markers",
                    "signal_understanding_markers",
                    "signal_code_indicators",
                ]
            feature_names.extend(convo_feats)
    return feature_names


def explain_logistic_regression(pipeline: Pipeline, top_k: int = 8) -> Dict[str, Any]:
    """
    Extracts class-wise positive and negative feature weights from Logistic Regression.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]
    feature_names = get_pipeline_feature_names(preprocessor)
    classes = clf.classes_.tolist()
    coef = clf.coef_  # Shape: (n_classes, n_features)

    explanations = {}
    for idx, cls_name in enumerate(classes):
        weights = coef[idx]
        sorted_indices = np.argsort(weights)

        # Top positive features (increase probability of this class)
        top_pos_idx = sorted_indices[-top_k:][::-1]
        top_pos = [
            {"feature": feature_names[i], "weight": round(float(weights[i]), 4)}
            for i in top_pos_idx if weights[i] > 0
        ]

        # Top negative features (decrease probability of this class)
        top_neg_idx = sorted_indices[:top_k]
        top_neg = [
            {"feature": feature_names[i], "weight": round(float(weights[i]), 4)}
            for i in top_neg_idx if weights[i] < 0
        ]

        explanations[cls_name] = {
            "top_positive_features": top_pos,
            "top_negative_features": top_neg,
        }

    return {
        "model_type": "LogisticRegression",
        "classes": classes,
        "class_explanations": explanations,
    }


def explain_linear_svm(pipeline: Pipeline, top_k: int = 8) -> Dict[str, Any]:
    """
    Extracts decision hyperplane weights from Linear SVM (or CalibratedClassifierCV wrapper).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]
    feature_names = get_pipeline_feature_names(preprocessor)
    classes = getattr(clf, "classes_", ["confused", "partial", "understood"]).tolist()

    # Extract base LinearSVC coefficients
    if hasattr(clf, "calibrated_classifiers_"):
        # Average coefficients across calibrated folds
        base_coefs = [cc.estimator.coef_ for cc in clf.calibrated_classifiers_]
        coef = np.mean(base_coefs, axis=0)
    elif hasattr(clf, "coef_"):
        coef = clf.coef_
    else:
        return {"error": "Unable to extract coefficients from classifier."}

    explanations = {}
    for idx, cls_name in enumerate(classes):
        weights = coef[idx]
        sorted_indices = np.argsort(weights)

        top_pos_idx = sorted_indices[-top_k:][::-1]
        top_pos = [
            {"feature": feature_names[i], "weight": round(float(weights[i]), 4)}
            for i in top_pos_idx if weights[i] > 0
        ]

        top_neg_idx = sorted_indices[:top_k]
        top_neg = [
            {"feature": feature_names[i], "weight": round(float(weights[i]), 4)}
            for i in top_neg_idx if weights[i] < 0
        ]

        explanations[cls_name] = {
            "top_positive_features": top_pos,
            "top_negative_features": top_neg,
        }

    return {
        "model_type": "LinearSVC (Calibrated)",
        "classes": classes,
        "class_explanations": explanations,
    }


def explain_naive_bayes(pipeline: Pipeline, top_k: int = 8) -> Dict[str, Any]:
    """
    Extracts most distinctive features per class using relative log-likelihood ratios.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]
    feature_names = get_pipeline_feature_names(preprocessor)
    classes = clf.classes_.tolist()
    log_prob = clf.feature_log_prob_  # Shape: (n_classes, n_features)

    explanations = {}
    for idx, cls_name in enumerate(classes):
        # Compute relative log-likelihood ratio against other classes
        other_mean = np.mean([log_prob[j] for j in range(len(classes)) if j != idx], axis=0)
        diff = log_prob[idx] - other_mean
        top_idx = np.argsort(diff)[-top_k:][::-1]

        distinctive = [
            {
                "feature": feature_names[i],
                "log_prob_class": round(float(log_prob[idx][i]), 4),
                "distinctiveness_ratio": round(float(diff[i]), 4),
            }
            for i in top_idx
        ]
        explanations[cls_name] = {
            "most_distinctive_terms": distinctive,
        }

    return {
        "model_type": "MultinomialNB",
        "classes": classes,
        "class_explanations": explanations,
    }


def explain_quiz_regressor(pipeline: Pipeline, top_k: int = 10) -> Dict[str, Any]:
    """
    Extracts top contributing features for the Quiz Score Regressor.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    reg = pipeline.named_steps["regressor"]
    feature_names = get_pipeline_feature_names(preprocessor)

    if hasattr(reg, "feature_importances_"):
        importances = reg.feature_importances_
        top_indices = np.argsort(importances)[-top_k:][::-1]
        ranking = [
            {"feature": feature_names[i], "importance": round(float(importances[i]), 4)}
            for i in top_indices
        ]
        method = "RandomForest Gini / Variance Reduction"
    elif hasattr(reg, "coef_"):
        coef = reg.coef_
        top_indices = np.argsort(np.abs(coef))[-top_k:][::-1]
        ranking = [
            {"feature": feature_names[i], "coefficient": round(float(coef[i]), 4)}
            for i in top_indices
        ]
        method = "Ridge Regression L2 Coefficients"
    else:
        return {"error": "Unknown regressor type."}

    return {
        "model_type": reg.__class__.__name__,
        "importance_method": method,
        "top_features": ranking,
    }


def explain_instance(
    pipeline: Pipeline,
    conversation_input: Union[Dict[str, Any], List[str], str],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Computes local feature attribution for a single student conversation.
    Decomposes the prediction into the specific linguistic and conversational signals driving it.
    """
    if isinstance(conversation_input, dict):
        student_msgs = extract_student_messages(conversation_input)
        if not student_msgs and "full_text" in conversation_input:
            student_msgs = [conversation_input["full_text"]]
        elif not student_msgs and "text" in conversation_input:
            student_msgs = [conversation_input["text"]]
    elif isinstance(conversation_input, list):
        student_msgs = [str(m).strip() for m in conversation_input if str(m).strip()]
    elif isinstance(conversation_input, str):
        lines = [l.strip() for l in conversation_input.strip().split("\n") if l.strip()]
        student_msgs = lines
    else:
        student_msgs = []

    if not student_msgs:
        return {"error": "No student messages provided for explanation."}

    full_text = " ".join(student_msgs)
    input_df = pd.DataFrame([{"full_text": full_text, "student_messages": student_msgs}])

    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]
    feature_names = get_pipeline_feature_names(preprocessor)
    X_trans = preprocessor.transform(input_df)
    if hasattr(X_trans, "toarray"):
        x_vec = X_trans.toarray()[0]
    else:
        x_vec = np.asarray(X_trans)[0]

    predicted_class = str(pipeline.predict(input_df)[0])
    classes = clf.classes_.tolist()
    class_idx = classes.index(predicted_class)

    # Extract weights for the predicted class
    if hasattr(clf, "coef_"):
        weights = clf.coef_[class_idx]
    elif hasattr(clf, "calibrated_classifiers_"):
        base_coefs = [cc.estimator.coef_ for cc in clf.calibrated_classifiers_]
        weights = np.mean(base_coefs, axis=0)[class_idx]
    else:
        weights = np.ones_like(x_vec)

    # Contribution = weight * transformed_feature_value
    contributions = weights * x_vec
    active_indices = [i for i in range(len(x_vec)) if abs(x_vec[i]) > 1e-4]

    # Sort active features by magnitude of contribution
    top_pos_idx = sorted(active_indices, key=lambda i: contributions[i], reverse=True)[:top_k]
    top_neg_idx = sorted(active_indices, key=lambda i: contributions[i])[:top_k]

    pos_drivers = [
        {
            "feature": feature_names[i],
            "value": round(float(x_vec[i]), 3),
            "weight": round(float(weights[i]), 4),
            "contribution": round(float(contributions[i]), 4),
        }
        for i in top_pos_idx if contributions[i] > 0
    ]

    neg_drivers = [
        {
            "feature": feature_names[i],
            "value": round(float(x_vec[i]), 3),
            "weight": round(float(weights[i]), 4),
            "contribution": round(float(contributions[i]), 4),
        }
        for i in top_neg_idx if contributions[i] < 0
    ]

    summary_text = (
        f"Prediction '{predicted_class.upper()}' is driven primarily by positive contributions from "
        f"{', '.join([d['feature'] for d in pos_drivers[:3]]) if pos_drivers else 'overall context'}."
    )

    return {
        "predicted_class": predicted_class,
        "positive_evidence_drivers": pos_drivers,
        "negative_evidence_drivers": neg_drivers,
        "top_positive_features": pos_drivers,
        "top_negative_features": neg_drivers,
        "total_active_features": len(active_indices),
        "summary": summary_text,
    }


def generate_explainability_report(reports_dir: Path = ROOT_DIR / "reports") -> Dict[str, Any]:
    """Generates and saves global explainability report across all models."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    # Logistic Regression
    lr_path = ROOT_DIR / "models" / "understanding" / "logistic_regression" / "model.joblib"
    if lr_path.exists():
        pipe_lr = joblib.load(lr_path)
        report["logistic_regression"] = explain_logistic_regression(pipe_lr)

    # Linear SVM
    svm_path = ROOT_DIR / "models" / "understanding" / "linear_svm" / "model.joblib"
    if svm_path.exists():
        pipe_svm = joblib.load(svm_path)
        report["linear_svm"] = explain_linear_svm(pipe_svm)

    # Naive Bayes
    nb_path = ROOT_DIR / "models" / "understanding" / "naive_bayes" / "model.joblib"
    if nb_path.exists():
        pipe_nb = joblib.load(nb_path)
        report["naive_bayes"] = explain_naive_bayes(pipe_nb)

    # Quiz Regressor
    quiz_path = ROOT_DIR / "models" / "quiz_score" / "model.joblib"
    if quiz_path.exists():
        pipe_quiz = joblib.load(quiz_path)
        report["quiz_regressor"] = explain_quiz_regressor(pipe_quiz)

    out_file = reports_dir / "model_explainability.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved explainability report to: {out_file}")

    return report


if __name__ == "__main__":
    generate_explainability_report()
