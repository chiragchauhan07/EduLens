# EduLens — Student Learning Intelligence

> **A focused, lightweight Machine Learning application for analyzing educational conversations, predicting student topic comprehension, and estimating expressed learning confidence.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)]()

---

## 1. Project Introduction

**EduLens** is a specialized Machine Learning application built for educational learning analytics. 

Instead of functioning as an AI tutor or conversational chatbot, EduLens acts as an **independent analytical lens**. It takes an educational conversation between a student and an AI tutor (or human teacher), extracts interpretable linguistic and conversational signals, and predicts:

1. **Student Topic Understanding**: Classified as <span style="color: #059669; font-weight: bold;">Understood</span>, <span style="color: #d97706; font-weight: bold;">Partially Understood</span>, or <span style="color: #e11d48; font-weight: bold;">Confused</span> using a supervised Machine Learning classifier.
2. **Expressed Learning Confidence**: Estimated as **High Confidence**, **Moderate Confidence**, or **Low Confidence** using a transparent, rule-calibrated linguistic marker model based on hedges, assertions, and self-repair cues.

EduLens runs completely locally without requiring external LLM API keys or cloud dependencies during inference.

---

## 2. Problem Statement & Objectives

### Problem Statement
When students learn with external AI models (such as ChatGPT, Claude, or Socratic tutors), the dialogue transcripts contain rich evidence of comprehension, misconceptions, and hesitation. However, teachers, institutions, and students lack lightweight, transparent tools to evaluate whether the student genuinely understood the material or is merely repeating answers with unresolved confusion.

### Project Objectives
- **Automated Dialogue Assessment**: Process multi-turn student-tutor dialogues in standard JSON format.
- **Interpretable NLP Signal Extraction**: Combine vocabulary n-grams (TF-IDF) with conversational markers (questions, confusion words, affirmations, code syntax).
- **High-Accuracy Classification**: Deploy a trained Scikit-Learn classifier with zero student-persona leakage.
- **Transparent Confidence Analysis**: Provide explicit linguistic confidence estimation distinct from classifier probabilities.
- **Minimal, Modern Interface**: Serve an intuitive Apple/Google-inspired web dashboard designed with Google Stitch MCP.

---

## 3. System Architecture

```text
Student Uploads JSON Conversation
              │
              ▼
    [ Conversation Parser ]
      ├─ Validates JSON syntax & file size (max 5MB)
      ├─ Extracts Topic, Student Turns & Assistant Turns
      └─ Rejects empty or teacher-only dialogues
              │
              ▼
    [ NLP & Feature Pipeline ]
      ├─ Normalizes & cleans utterance text
      ├─ 100 Lexical TF-IDF N-grams (1, 2)
      └─ 8 Conversational & Pedagogical Signals
              │
              ▼
       (108-Dimensional Vector)
              │
     ┌────────┴──────────────────────────┐
     ▼                                   ▼
[ ML Understanding Classifier ]    [ Linguistic Confidence Estimator ]
  ├─ Multinomial Logistic Reg.       ├─ Uncertainty & Hedge density
  ├─ 95.89% Test Accuracy            ├─ Assertion & Certainty tokens
  └─ Calibrated Softmax Probs        └─ Metacognitive self-repair score
     │                                   │
     └────────────────┬──────────────────┘
                      ▼
          [ FastAPI Analysis Route ]
                      ▼
         [ EduLens Web Dashboard ]
           (Apple / Google Minimalist UI)
```

---

## 4. Technology Stack

| Layer | Technologies |
|---|---|
| **Backend & API** | Python 3.12, FastAPI, Uvicorn, Pydantic |
| **Machine Learning** | Scikit-Learn 1.5.2, Joblib, NumPy, Pandas |
| **NLP & Signal Processing** | TF-IDF Vectorization, Regex Token Extractors, StandardScaler |
| **Frontend UI** | HTML5, Tailwind CSS, Vanilla JavaScript, Google Stitch MCP Design |
| **Testing & Quality** | Pytest, Starlette TestClient (57 automated tests passing) |

---

## 5. Machine Learning & NLP Architecture

### A. Feature Extraction (108 Total Dimensions)
The inference pipeline applies a Scikit-Learn `ColumnTransformer` identical to the training specification:

1. **Lexical Subspace (100 Features)**:
   - `TfidfVectorizer(max_features=100, ngram_range=(1, 2), min_df=2, stop_words='english')`
   - Captures domain vocabulary, causal conjunctions (*because*, *therefore*, *which means*), and conceptual terminology.

2. **Conversational Subspace (8 Features)**:
   - `ConversationalFeatureExtractor()` followed by `StandardScaler()`:
     1. `num_turns`: Number of student turns.
     2. `total_words`: Total student word count.
     3. `avg_turn_length`: Average words per student utterance.
     4. `question_count`: Count of question marks (`?`).
     5. `confusion_markers`: Matches regex for doubt, stuck, unclear, don't get, blank.
     6. `clarification_markers`: Matches requests for examples, repeat, step-by-step.
     7. `understanding_markers`: Matches affirmations (*got it*, *makes sense*, *understood*).
     8. `code_indicators`: Matches syntax markers, backticks, programming keywords.

### B. Understanding Classifier
- **Algorithm**: Multinomial Logistic Regression (`class_weight='balanced'`, `max_iter=1000`, `C=1.0`)
- **Classes**:
  - **`Understood`**: Student articulates concepts clearly, provides explanations, and exhibits high comprehension.
  - **`Partially Understood`**: Student shows fundamental grasp but has lingering questions, asks for clarification, or struggles with edge cases.
  - **`Confused`**: Student expresses explicit doubt, confusion, incorrect explanations, or instructional deadlock.

### C. Expressed Confidence Estimation (Linguistic Evidence)
- **Methodology**: Rule-calibrated conversational discourse analysis.
- **Signals Extracted**:
  - `uncertainty_count`: Hedges (*not sure*, *maybe*, *i guess*, *idk*, *i don't know*).
  - `certainty_count`: Assertive markers (*definitely*, *clearly*, *i am certain*, *obviously*).
  - `hesitation_count`: Hesitation tokens (*um*, *uh*, *hmm*, ellipses `...`).
  - `self_corrections`: Metacognitive adjustments (*oh wait*, *actually it's*, *i meant*).
  - `assertion_hedge_ratio`: Ratio of confident to hesitant markers.
- **Score**: Bounded between 5.0% and 95.0%.
- **Output Levels**: **High Confidence** (> 68%), **Moderate Confidence** (40%–68%), **Low Confidence** (< 40%).
- *Important*: This is explicitly labeled as a **linguistic confidence estimate**, not an internal psychometric measurement.

---

## 6. Model Evaluation Results

Evaluated on the primary supervised learning dataset using **5-fold Stratified Group K-Fold Cross-Validation** grouped by `student_id` to guarantee **zero persona leakage**:

| Metric | Result |
|---|---|
| **Test Accuracy** | **95.89%** |
| **Macro Precision** | **0.9380** |
| **Macro Recall** | **0.9429** |
| **Macro F1-Score** | **0.9399** |
| **Weighted F1-Score** | **0.9591** |
| **5-Fold CV Accuracy** | **93.78% (± 3.2%)** |

### Confusion Matrix (Test Split)
```text
                  Predicted: Confused   Partial   Understood
Actual: Confused            13             1           0
Actual: Partial              2            18           0
Actual: Understood           0             0          39
```

---

## 7. Supported JSON Conversation Formats

EduLens supports any standard educational dialogue format.

### Recommended Format
```json
{
  "topic": "Binary Search",
  "messages": [
    {
      "role": "assistant",
      "content": "Can you explain how binary search finds an element?"
    },
    {
      "role": "student",
      "content": "It divides the sorted array in half repeatedly until the target is found."
    },
    {
      "role": "assistant",
      "content": "Correct! What is its time complexity?"
    },
    {
      "role": "student",
      "content": "Got it, that makes complete sense! It runs in O(log n) time. Thanks!"
    }
  ]
}
```

### Alternative Supported Schemas
- `turns` array containing `student_message` and `assistant_message` (legacy EduLens dataset format).
- `student_turns` and `tutor_turns` separate arrays.
- `role` / `speaker` variations: `"student"`, `"learner"`, `"user"` vs `"assistant"`, `"ai"`, `"tutor"`, `"teacher"`.
- Text field variations: `"content"`, `"text"`, `"message"`, `"utterance"`.

---

## 8. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Windows, macOS, or Linux

### 1. Clone & Navigate
```powershell
git clone https://github.com/chiragchauhan07/EduLens.git
cd EduLens
```

### 2. Set Up Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 9. Launching the Application

Start the local server with one command:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at:
**[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

The application serves the clean Apple/Google-inspired web dashboard directly through FastAPI.

---

## 10. Automated Test Suite

Run the full automated test suite verifying JSON parsing, validation, ML pipeline, and API endpoints:

```powershell
pytest tests/ -v
```

Expected output:
```text
======================= 57 passed, 2 warnings in 7.80s ========================
```

---

## 11. Evaluation Test Cases

Representative test JSON files are provided in the [`samples/`](file:///samples) directory:

| Sample File | Scenario | Expected Understanding | Expected Confidence |
|---|---|---|---|
| [`case1_understood.json`](file:///samples/case1_understood.json) | Student articulates binary search trees clearly | **Understood** (99.9%) | **High Confidence** (71.0%) |
| [`case2_partial_understanding.json`](file:///samples/case2_partial_understanding.json) | Student grasps recursion with unwinding hesitation | **Understood / Partial** (59.9%) | **Moderate Confidence** (62.0%) |
| [`case3_confused.json`](file:///samples/case3_confused.json) | Student expresses deadlock with C pointers | **Confused** (97.9%) | **Moderate Confidence** (55.0%) |
| [`case4_low_confidence.json`](file:///samples/case4_low_confidence.json) | Student expresses heavy doubts and hedges | **Confused** (96.8%) | **Low Confidence** (5.0%) |
| [`case5_high_confidence.json`](file:///samples/case5_high_confidence.json) | Student asserts kinematics formulas with certainty | Output from Model | **High Confidence** (95.0%) |
| [`case6_invalid_syntax.json`](file:///samples/case6_invalid_syntax.json) | Syntax error in JSON payload | Rejection (**400 Invalid JSON**) | N/A |
| [`case7_empty.json`](file:///samples/case7_empty.json) | Conversation with empty message list | Rejection (**400 No Student Messages**) | N/A |
| [`case8_assistant_only.json`](file:///samples/case8_assistant_only.json) | Teacher monologue with zero student utterances | Rejection (**400 Only Assistant Messages**) | N/A |

---

## 12. Project Structure

```text
EduLens/
├── backend/
│   ├── main.py                     # FastAPI application entrypoint & static mount
│   ├── routes/
│   │   └── analysis.py             # Health, model-info, sample, and analyze endpoints
│   ├── services/
│   │   ├── conversation_parser.py  # Robust multi-schema JSON parser & validator
│   │   ├── preprocessing.py        # Utterance normalization and DataFrame assembler
│   │   ├── feature_extraction.py   # Conversational signals & linguistic confidence
│   │   └── prediction.py           # Model singleton & analysis orchestrator
│   └── models/
│       ├── understanding_classifier.joblib  # Trained Scikit-Learn Pipeline
│       └── metadata.json           # Model specifications & CV metrics
├── frontend/
│   ├── index.html                  # Apple/Google minimalist UI (Stitch MCP)
│   └── assets/
│       ├── app.js                  # Frontend controller & dynamic rendering
│       └── styles.css              # Custom styling & animation tokens
├── public/                         # Mirrored static production distribution
├── samples/                        # 8 Representative test conversations
├── ml_prototype/                   # Core data loader, DSP, and training scripts
├── data/                           # Training datasets & normalized session archives
├── tests/                          # 57 Comprehensive unit and integration tests
├── requirements.txt                # Lean production dependencies
└── README.md                       # Documentation
```

---

## 13. Limitations & Academic Scope

1. **Dialogue Evidence Only**: EduLens evaluates what the student *expresses* in the provided text dialogue. It does not measure latent internal IQ, long-term memory, or psychometric self-efficacy.
2. **Vocabulary Domain**: The TF-IDF sub-pipeline was trained primarily on STEM and Computer Science topics (algorithms, data structures, recursion, pointers). Non-STEM topics will rely primarily on the 8 conversational signals.
3. **Linguistic Confidence vs. True Mastery**: Expressed confidence reflects conversational assertiveness and hedging, not necessarily factual correctness. A student can sound confident while asserting a misconception.

---

## 14. License
MIT License. Developed for Student Learning Intelligence.
