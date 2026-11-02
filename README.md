# EduLens

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](#running-tests)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Prototype%20Delivered-success.svg)](#project-status)

**NLP and Machine Learning-Based Student Learning Intelligence and Performance Analytics Platform**

*EduLens evaluates demonstrated understanding, diagnoses conceptual confusion, and tracks curriculum mastery from student–LLM learning dialogues.*

</div>

---

## 📌 What is EduLens?

**EduLens is NOT another AI tutor.** It does not teach students or answer conversational questions.

Today, students already learn complex subjects using external AI chatbots (such as **ChatGPT**, **Claude**, or **Gemini**). However, conversational chatbots have a critical blind spot: **they cannot objectively measure whether real learning occurred.**

EduLens serves as the **intelligence and analytics layer** built around the student's existing learning habits:
1. Students complete a study session on any topic with an external LLM.
2. They share their chat session with EduLens (via file upload, API, or Model Context Protocol).
3. EduLens extracts verifiable, interpretable conversational evidence and linguistic features.
4. An explainable Machine Learning pipeline predicts the student's demonstrated level of understanding (`understood`, `partial`, or `confused`).
5. A comprehensive analytics dashboard provides curriculum-level topic mastery breakdown, struggle identification, and targeted pedagogical feedback.

```
┌─────────────────────────────────────────────────────────────┐
│               Student Learns via External AI                │
│             (ChatGPT / Claude / Gemini / IDE)               │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Session Log Ingestion                       │
│    (JSON Transcripts / Model Context Protocol (MCP))        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            NLP & Feature Extraction Pipeline                │
│    (8 Conversational Signals + 100 Lexical TF-IDF N-Grams)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Interpretable ML Classifier                     │
│    (Multinomial Logistic Regression with Balanced Weights)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│           EduLens Analytics & Feedback Dashboard            │
│  - Real-Time Understanding Prediction & Confidence Scores   │
│  - Curriculum Topic Mastery & Confusion Analytics           │
│  - Formative Pedagogical Guidance for Next Study Steps       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Features

- **🔍 Multi-Schema Session Ingestion:** Accepts conversation logs across standard formats (`turns`, `messages`, raw dialogue, or JSON files).
- **🧪 Leakage-Free Feature Engineering:** 108 features combining 8 dense pedagogical & structural signals (turn counts, word ratios, confusion markers, clarification requests, code indicators) with 100 TF-IDF lexical n-grams.
- **🎯 High-Precision Understanding Classifier:** Categorizes student comprehension into `understood`, `partial`, and `confused` with **96.51% test accuracy** and **0.9577 macro F1-score** (vs. 45.35% baseline).
- **📊 Interactive Streamlit Analytics Dashboard:**
  - **Student Performance Analyzer:** Test real-time inference on sample dialogues, paste custom transcripts, or upload files.
  - **Topic-Wise Performance:** Curriculum-level mastery aggregates, confusion rates, and topic difficulty rankings.
  - **Model Architecture & Viva Guide:** Technical documentation, feature weights, confusion matrix, and defense Q&A.
- **⚡ Deterministic & Explainable:** Sub-millisecond inference per session with transparent probability distributions and explainable linear coefficients.
- **🛡️ Automated Test Suite:** Unit and integration tests powered by `pytest` ensuring end-to-end pipeline reliability.

---

## 🗂️ Repository Structure

```text
EduLens/
├── .gitignore                             # Git ignore rules (caches, venvs, env files)
├── pytest.ini                             # Pytest configuration (pythonpath = .)
├── requirements.txt                       # Production & prototype dependencies
├── EduLens Project Specification.pdf     # System specification & architectural blueprint
├── PROJECT_CONTEXT.md                     # High-level architecture, roadmap & vision
├── prototype_context.md                   # ML prototype details & viva defense reference
├── README.md                              # Main onboarding guide (this file)
├── generate_dataset.py                    # Synthetic CS student dialogue generator
├── data/
│   └── raw/
│       ├── edulens_chats_dataset.json     # Primary supervised dataset (430 sessions)
│       ├── edulens_chats_set1 (1).json    # Supplementary dataset (420 sessions)
│       ├── edulens_chats_set2 (1).json    # Supplementary dataset (430 sessions)
│       ├── edulens_chats_set3 (1).json    # Supplementary dataset (440 sessions)
│       ├── education_dialogue/            # 47k Gemini dialogues (unlabeled)
│       └── mathdial/                      # 2.8k human math tutoring dialogues
├── ml_prototype/                          # Core Machine Learning Package
│   ├── __init__.py                        # Package root
│   ├── data_loader.py                     # Schema normalization & dataset loader
│   ├── features.py                        # Conversational signals & TF-IDF transformers
│   ├── train_model.py                     # Stratified training, 5-fold CV & evaluation
│   ├── predict.py                         # Inference engine with confidence & feedback
│   └── dashboard.py                       # 3-page interactive Streamlit dashboard
├── models/
│   └── trained_model.joblib               # Serialized Scikit-learn Pipeline artifact
├── reports/
│   ├── dataset_inspection.md              # Comprehensive dataset schema & audit report
│   └── model_evaluation.json              # Model evaluation metrics & confusion matrix
└── tests/
    ├── __init__.py
    └── test_prototype.py                  # Pytest automated test suite
```

---

## 📊 Machine Learning Architecture

### Prediction Target
- **`final_understanding`:** 3-class categorical label representing demonstrated conceptual mastery:
  - **`understood`** (44.7%): Student articulated concepts correctly, resolved questions, and confirmed clarity.
  - **`partial`** (36.3%): Student understood core ideas but required repeated clarifications or exhibited hesitation.
  - **`confused`** (19.1%): Student expressed persistent difficulty, repeated doubts, or failed to apply concepts.

### Feature Pipeline (108 Features)
1. **Dense Conversational Signals (8 Features, Scaled via StandardScaler):**
   - `num_turns`: Count of student messages.
   - `total_words`: Total word count across student utterances.
   - `avg_turn_length`: Average words per turn.
   - `question_count`: Frequency of question marks (`?`).
   - `confusion_markers`: Regex occurrences of confusion phrases (*lost, stuck, don't understand, doubt*).
   - `clarification_markers`: Regex occurrences of clarification requests (*explain again, another example*).
   - `understanding_markers`: Regex occurrences of comprehension confirmations (*got it, makes sense, clear now*).
   - `code_indicators`: Count of code syntax elements (*backticks, def, class, return, brackets*).
2. **Lexical Vocabulary (100 Features):**
   - `TfidfVectorizer(max_features=100, ngram_range=(1, 2), min_df=2, stop_words="english")` fit exclusively on student text.

### Model Evaluation Results

Evaluated on held-out test split (20%, 86 sessions) with Stratified 5-Fold Cross Validation:

| Metric | EduLens Primary Model | Baseline (Majority Class) |
|---|---|---|
| **Model Type** | **Multinomial Logistic Regression** | `DummyClassifier` |
| **Test Accuracy** | **96.51%** | 45.35% |
| **Macro Precision** | **0.9708** | 0.1512 |
| **Macro Recall** | **0.9476** | 0.3333 |
| **Macro F1-Score** | **0.9577** | 0.2080 |
| **Weighted F1-Score** | **0.9647** | 0.2833 |
| **5-Fold CV Accuracy** | **91.88% (± 4.26%)** | — |
| **5-Fold CV Macro F1** | **90.31% (± 5.06%)** | — |

#### Confusion Matrix (Held-Out Test Set)
```text
                  Predicted: confused  partial  understood
Actual: confused                14        2           0
Actual: partial                  0       30           1
Actual: understood               0        0          39
```

---

## 🛠️ Getting Started (For Team Members)

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Clone Repository
```powershell
git clone https://github.com/chiragchauhan07/EduLens.git
cd EduLens
```

### 3. Setup Virtual Environment
```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux (Bash)
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 💻 Running the Project

### Train & Evaluate the ML Model
To retrain the model and regenerate `models/trained_model.joblib` and `reports/model_evaluation.json`:
```powershell
python ml_prototype/train_model.py
```

### Run Real-Time Inference (CLI Test)
Test the inference engine on sample conversations:
```powershell
python ml_prototype/predict.py
```

### Run Automated Tests
Verify code integrity with pytest:
```powershell
pytest
```

### Launch Interactive Streamlit Dashboard
```powershell
streamlit run ml_prototype/dashboard.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ Streamlit Dashboard Walkthrough

The web dashboard is organized into three dedicated modules:

1. **🎓 Student Performance Analyzer:**
   - Load pre-built sample chats (Pointers, Recursion, Dynamic Programming, SQL, etc.).
   - Type or paste arbitrary student conversation transcripts.
   - Upload custom session JSON files.
   - Inspect predicted understanding level, class probabilities, confidence score, and extracted conversational signals in real time.

2. **📈 Topic-Wise Performance:**
   - Curriculum-level aggregation across 11 CS subjects and 40+ topics.
   - Drill-down tables displaying Mastery Rate (%), Partial Understanding Rate (%), and Confusion Rate (%).
   - Identify high-struggle topics needing reinforcement.

3. **ℹ️ Model Information & Viva Guide:**
   - Full technical breakdown of features and model parameters.
   - Baseline comparison tables and held-out test metrics.
   - Interactive viva defense guide addressing common architectural and design questions.

---

## 🗺️ Engineering Roadmap

- [x] **Phase 1: ML Prototype & Diagnostics** — Curated dataset, leakage-free feature extractor, 96.5% accurate model, Streamlit UI.
- [ ] **Phase 2: Live Ingestion & MCP Integration** — Model Context Protocol server for direct submission from Claude Desktop and ChatGPT.
- [ ] **Phase 3: Targeted Quiz Generation** — Gemini API integration to synthesize adaptive 10–15 question validation quizzes.
- [ ] **Phase 4: Longitudinal Knowledge Tracing** — Multi-session tracking per student across academic terms.
- [ ] **Phase 5: Production Web Platform** — Next.js educator dashboard with FastAPI backend and PostgreSQL persistence.

---

## 🔒 Data Privacy & Ethics

Student conversations are sensitive learning records:
- **Explicit Consent Only:** EduLens never scrapes or automatically captures chats without student initiation.
- **Privacy Boundary:** Student text is isolated from external third parties; model inference runs locally and deterministically.
- **No Hallucinated Evaluations:** Model classifications are grounded in verified conversational linguistic markers with calibrated confidence scores.

---

## 👥 Contributors

- **Chirag Chauhan** — Project Lead & Developer ([@chiragchauhan07](https://github.com/chiragchauhan07))
- **EduLens Engineering Team**

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
