# Prediction Walkthrough — End-to-End Pipeline Trace

This document demonstrates the deterministic progression of a student learning conversation through every stage of the EduLens Data Science, NLP, and Machine Learning pipeline.

---

## Stage 1: Raw JSON Conversation Ingestion

EduLens receives learning dialogues in structured JSON format via the Model Context Protocol (MCP) or direct API payload.

```json
{
  "chat_id": "CHAT_0001",
  "student_id": "S06",
  "subject": "C++",
  "topic": "Pointers",
  "student_persona": "exam",
  "turns": [
    {
      "turn": 1,
      "student_message": "explain pointers and references in C++ the way it is asked in university exams please ?"
    },
    {
      "turn": 2,
      "student_message": "can you show a quick dry run for this snippet ? i get confused tracing values ."
    },
    {
      "turn": 3,
      "student_message": "wait , why did the value change here ? what is happening with the address ? i am lost ."
    },
    {
      "turn": 4,
      "student_message": "is there a simpler way to remember this rule for midterms ? please repeat ."
    },
    {
      "turn": 5,
      "student_message": "thank you , will revise this again before the lab test tomorrow !"
    }
  ]
}
```

---

## Stage 2: Student Utterance Extraction

The extraction module (`extract_student_messages`) isolates student dialogue turns and strips away metadata, timestamps, and instructor responses to prevent target leakage:

```text
Turn 1: explain pointers and references in C++ the way it is asked in university exams please ?
Turn 2: can you show a quick dry run for this snippet ? i get confused tracing values .
Turn 3: wait , why did the value change here ? what is happening with the address ? i am lost .
Turn 4: is there a simpler way to remember this rule for midterms ? please repeat .
Turn 5: thank you , will revise this again before the lab test tomorrow !
```

---

## Stage 3: Text Normalization & Assembly

Conversational turns are normalized and concatenated into a single session transcript for lexical processing:

> *"explain pointers and references in C++ the way it is asked in university exams please ? can you show a quick dry run for this snippet ? i get confused tracing values . wait , why did the value change here ? what is happening with the address ? i am lost . is there a simpler way to remember this rule for midterms ? please repeat . thank you , will revise this again before the lab test tomorrow !"*

- **Character Count:** 397
- **Total Word Count:** 81

---

## Stage 4: Dense Conversational Feature Extraction (8 Signals)

Eight pedagogical and structural features are extracted via regular expression pattern matchers:

| Metric Name | Value | Pedagogical Description |
|---|---|---|
| `num_turns` | 5.0 | Total dialogue turns engaged by student |
| `total_words` | 81.0 | Aggregate words articulated |
| `avg_turn_length` | 16.20 | Average words per turn |
| `question_count` | 5.0 | Number of explicit question marks (?) |
| `confusion_markers` | 2.0 | Matches for phrases like "confused", "lost" |
| `clarification_markers` | 2.0 | Matches for phrases like "explain again", "repeat" |
| `understanding_markers` | 1.0 | Matches for affirmative comprehension |
| `code_indicators` | 2.0 | Programming syntax or language keyword tokens |

---

## Stage 5: Numerical Feature Vectorization (108 Dimensions)

The Scikit-learn `ColumnTransformer` applies:
1. `TfidfVectorizer(max_features=100, ngram_range=(1, 2))` on `full_text`.
2. `StandardScaler()` on the 8 dense conversational signals.

- **Total Dimensions:** 108 features.
- **Active Non-Zero Features:** 18
- **Sample Non-Zero Values:**
```json
[
  [
    "tfidf_dry",
    0.2416
  ],
  [
    "tfidf_dry run",
    0.2416
  ],
  [
    "tfidf_explain",
    0.1814
  ],
  [
    "tfidf_lost",
    0.2778
  ],
  [
    "tfidf_quick",
    0.2348
  ],
  [
    "tfidf_quick dry",
    0.2416
  ],
  [
    "tfidf_run",
    0.237
  ],
  [
    "tfidf_thank",
    0.2697
  ]
]
```

---

## Stage 6: Machine Learning Classification

The feature vector $\mathbf{x} \in \mathbb{R}^{108}$ is evaluated by the primary calibrated classifier:

- **Predicted Class:** **`understood`**
- **Calibrated Probabilities:**
  - `confused`: **16.44%**
  - `partial`: **29.99%**
  - `understood`: **53.57%**
- **Model Confidence:** **53.57%**

---

## Stage 7: Post-Session Quiz Score Regression

The Random Forest Regressor estimates the expected post-learning quiz score:

- **Predicted Quiz Score:** **59.3%**

---

## Stage 8: Local Feature Attribution (Explainability)

Local contribution analysis reveals the specific signals that drove the `understood` prediction:

### Positive Evidence Drivers:
```json
[
  {
    "feature": "tfidf_thank",
    "value": 0.27,
    "weight": 1.3609,
    "contribution": 0.367
  },
  {
    "feature": "signal_confusion_markers",
    "value": 1.511,
    "weight": 0.1963,
    "contribution": 0.2966
  },
  {
    "feature": "signal_question_count",
    "value": 0.728,
    "weight": 0.2932,
    "contribution": 0.2135
  },
  {
    "feature": "signal_num_turns",
    "value": -0.215,
    "weight": -0.3023,
    "contribution": 0.0649
  }
]
```

### Negative Evidence Drivers:
```json
[
  {
    "feature": "signal_clarification_markers",
    "value": 2.207,
    "weight": -0.2035,
    "contribution": -0.449
  },
  {
    "feature": "signal_understanding_markers",
    "value": -0.119,
    "weight": 2.3114,
    "contribution": -0.2761
  },
  {
    "feature": "tfidf_explain",
    "value": 0.181,
    "weight": -0.3574,
    "contribution": -0.0648
  },
  {
    "feature": "signal_total_words",
    "value": -0.168,
    "weight": 0.2828,
    "contribution": -0.0475
  }
]
```

---

## Conclusion

This walkthrough demonstrates that EduLens evaluates comprehension through **multi-factor conversational evidence** rather than artificial metadata shortcuts.
