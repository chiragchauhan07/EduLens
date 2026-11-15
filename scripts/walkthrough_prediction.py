"""
scripts/walkthrough_prediction.py — Complete Step-by-Step Prediction Trace for EduLens.

Traces an actual conversation through every stage of the pipeline:
1. Raw JSON session input.
2. Student message extraction (stripping metadata and instructor text).
3. Text normalization and sequential concatenation.
4. Dense conversational feature extraction (8 structural and pedagogical signals).
5. TF-IDF vectorization (100 lexical n-gram features).
6. 108-dimensional numerical feature vector construction.
7. Machine Learning model inference (Logistic Regression, Linear SVM, Naive Bayes).
8. Post-session quiz score regression.
9. Final structured educational intelligence output.

Exports the trace to 'reports/prediction_walkthrough.md'.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml_prototype.data_loader import extract_student_messages, normalize_session, load_primary_dataset
from ml_prototype.features import extract_conversation_stats, build_feature_pipeline
from ml_prototype.predict import EduLensPredictor
from ml_prototype.explainability import explain_instance, get_pipeline_feature_names


def run_walkthrough() -> Dict[str, Any]:
    print("=" * 75)
    print("EDULENS STEP-BY-STEP PREDICTION WALKTHROUGH")
    print("=" * 75)

    # 1. Select a real sample from primary dataset
    df = load_primary_dataset()
    sample_record = df.iloc[0].to_dict()

    raw_json_sample = {
        "chat_id": sample_record["chat_id"],
        "student_id": sample_record["student_id"],
        "subject": sample_record["subject"],
        "topic": sample_record["topic"],
        "student_persona": sample_record["student_persona"],
        "turns": [
            {"turn": 1, "student_message": "explain pointers and references in C++ the way it is asked in university exams please ?"},
            {"turn": 2, "student_message": "can you show a quick dry run for this snippet ? i get confused tracing values ."},
            {"turn": 3, "student_message": "wait , why did the value change here ? what is happening with the address ? i am lost ."},
            {"turn": 4, "student_message": "is there a simpler way to remember this rule for midterms ? please repeat ."},
            {"turn": 5, "student_message": "thank you , will revise this again before the lab test tomorrow !"},
        ],
    }

    print("\n[Stage 1] Raw Input JSON Received:")
    print(f"  Chat ID: {raw_json_sample['chat_id']}")
    print(f"  Subject: {raw_json_sample['subject']} | Topic: {raw_json_sample['topic']}")
    print(f"  Turns Count: {len(raw_json_sample['turns'])}")

    # 2. Student Message Extraction
    extracted_msgs = extract_student_messages(raw_json_sample)
    print("\n[Stage 2] Student Message Extraction:")
    for i, msg in enumerate(extracted_msgs, 1):
        print(f"  Turn {i}: \"{msg}\"")

    # 3. Text Normalization
    full_text = " ".join(extracted_msgs)
    print("\n[Stage 3] Text Normalization & Assembly:")
    print(f"  Normalized text length: {len(full_text)} characters, {len(full_text.split())} words")
    print(f"  Preview: \"{full_text[:120]}...\"")

    # 4. Dense Conversational Feature Extraction (8 signals)
    stats = extract_conversation_stats(extracted_msgs)
    print("\n[Stage 4] Dense Conversational Signals (8 Metrics):")
    for k, v in stats.items():
        print(f"  - {k:<25}: {v}")

    # 5. Pipeline Preprocessing (108-Dimensional Feature Vector)
    pipe = joblib.load(ROOT_DIR / "models" / "trained_model.joblib")
    preprocessor = pipe.named_steps["preprocessor"]
    feature_names = get_pipeline_feature_names(preprocessor)

    input_df = pd.DataFrame([{"full_text": full_text, "student_messages": extracted_msgs}])
    X_trans = preprocessor.transform(input_df)
    if hasattr(X_trans, "toarray"):
        x_vec = X_trans.toarray()[0]
    else:
        x_vec = np.asarray(X_trans)[0]

    non_zero_features = [(feature_names[i], round(float(x_vec[i]), 4)) for i in range(len(x_vec)) if abs(x_vec[i]) > 1e-4]
    print("\n[Stage 5] Transformed Numerical Feature Vector:")
    print(f"  Vector Dimensionality: {len(x_vec)} features (100 TF-IDF + 8 Scaled Signals)")
    print(f"  Active (Non-Zero) Features: {len(non_zero_features)}")
    print(f"  Sample Active Features: {non_zero_features[:6]}")

    # 6. ML Model Inference
    pred_class = str(pipe.predict(input_df)[0])
    probs = pipe.predict_proba(input_df)[0]
    classes = pipe.classes_.tolist()
    prob_dict = {c: round(float(p), 4) for c, p in zip(classes, probs)}
    print("\n[Stage 6] ML Model Classification:")
    print(f"  Predicted Understanding Class: {pred_class.upper()}")
    print(f"  Calibrated Class Probabilities: {prob_dict}")

    # 7. Quiz Score Regression
    quiz_pipe = joblib.load(ROOT_DIR / "models" / "quiz_score" / "model.joblib")
    predicted_quiz = round(float(np.clip(quiz_pipe.predict(input_df)[0], 0.0, 100.0)), 1)
    print("\n[Stage 7] Quiz Score Regressor:")
    print(f"  Predicted Assessment Score: {predicted_quiz}%")

    # 8. Local Feature Attribution (Explainability)
    explanation = explain_instance(pipe, extracted_msgs, top_k=4)
    print("\n[Stage 8] Decision Driver Attribution:")
    print("  Top Positive Drivers (Pushed towards predicted class):")
    for d in explanation["positive_evidence_drivers"]:
        print(f"    + {d['feature']}: value={d['value']}, contribution={d['contribution']}")

    # 9. Generate Markdown Document
    md_content = f"""# Prediction Walkthrough — End-to-End Pipeline Trace

This document demonstrates the deterministic progression of a student learning conversation through every stage of the EduLens Data Science, NLP, and Machine Learning pipeline.

---

## Stage 1: Raw JSON Conversation Ingestion

EduLens receives learning dialogues in structured JSON format via the Model Context Protocol (MCP) or direct API payload.

```json
{json.dumps(raw_json_sample, indent=2)}
```

---

## Stage 2: Student Utterance Extraction

The extraction module (`extract_student_messages`) isolates student dialogue turns and strips away metadata, timestamps, and instructor responses to prevent target leakage:

```text
Turn 1: {extracted_msgs[0]}
Turn 2: {extracted_msgs[1]}
Turn 3: {extracted_msgs[2]}
Turn 4: {extracted_msgs[3]}
Turn 5: {extracted_msgs[4]}
```

---

## Stage 3: Text Normalization & Assembly

Conversational turns are normalized and concatenated into a single session transcript for lexical processing:

> *"{full_text}"*

- **Character Count:** {len(full_text)}
- **Total Word Count:** {len(full_text.split())}

---

## Stage 4: Dense Conversational Feature Extraction (8 Signals)

Eight pedagogical and structural features are extracted via regular expression pattern matchers:

| Metric Name | Value | Pedagogical Description |
|---|---|---|
| `num_turns` | {stats['num_turns']} | Total dialogue turns engaged by student |
| `total_words` | {stats['total_words']} | Aggregate words articulated |
| `avg_turn_length` | {stats['avg_turn_length']:.2f} | Average words per turn |
| `question_count` | {stats['question_count']} | Number of explicit question marks (?) |
| `confusion_markers` | {stats['confusion_markers']} | Matches for phrases like "confused", "lost" |
| `clarification_markers` | {stats['clarification_markers']} | Matches for phrases like "explain again", "repeat" |
| `understanding_markers` | {stats['understanding_markers']} | Matches for affirmative comprehension |
| `code_indicators` | {stats['code_indicators']} | Programming syntax or language keyword tokens |

---

## Stage 5: Numerical Feature Vectorization (108 Dimensions)

The Scikit-learn `ColumnTransformer` applies:
1. `TfidfVectorizer(max_features=100, ngram_range=(1, 2))` on `full_text`.
2. `StandardScaler()` on the 8 dense conversational signals.

- **Total Dimensions:** 108 features.
- **Active Non-Zero Features:** {len(non_zero_features)}
- **Sample Non-Zero Values:**
```json
{json.dumps(non_zero_features[:8], indent=2)}
```

---

## Stage 6: Machine Learning Classification

The feature vector $\mathbf{{x}} \in \mathbb{{R}}^{{108}}$ is evaluated by the primary calibrated classifier:

- **Predicted Class:** **`{pred_class}`**
- **Calibrated Probabilities:**
  - `confused`: **{prob_dict.get('confused', 0.0):.2%}**
  - `partial`: **{prob_dict.get('partial', 0.0):.2%}**
  - `understood`: **{prob_dict.get('understood', 0.0):.2%}**
- **Model Confidence:** **{max(prob_dict.values()):.2%}**

---

## Stage 7: Post-Session Quiz Score Regression

The Random Forest Regressor estimates the expected post-learning quiz score:

- **Predicted Quiz Score:** **{predicted_quiz}%**

---

## Stage 8: Local Feature Attribution (Explainability)

Local contribution analysis reveals the specific signals that drove the `{pred_class}` prediction:

### Positive Evidence Drivers:
```json
{json.dumps(explanation['positive_evidence_drivers'], indent=2)}
```

### Negative Evidence Drivers:
```json
{json.dumps(explanation['negative_evidence_drivers'], indent=2)}
```

---

## Conclusion

This walkthrough demonstrates that EduLens evaluates comprehension through **multi-factor conversational evidence** rather than artificial metadata shortcuts.
"""

    report_path = ROOT_DIR / "reports" / "prediction_walkthrough.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\nSaved prediction walkthrough report to: {report_path}")

    return {
        "chat_id": raw_json_sample["chat_id"],
        "predicted_class": pred_class,
        "probabilities": prob_dict,
        "predicted_quiz_score": predicted_quiz,
    }


if __name__ == "__main__":
    run_walkthrough()

