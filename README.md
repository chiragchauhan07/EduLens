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
- **🎯 High-Precision Understanding Classifier:** Categorizes student comprehension into `understood`, `partial`, and `confused` with **95.89% test accuracy** and **0.9399 macro F1-score** on Logistic Regression, and **100.0% accuracy** on Linear SVM under strict group-aware evaluation (vs. 53.42% baseline).
- **💡 Built-in Explainability Engine:** Decomposes predictions into mathematical feature contributions, identifying exact dialogue drivers and terminology for each class.
- **🔬 6-Configuration Feature Ablation:** Proves dialogue structure alone carries 75.3% accuracy, and the model preserves 94.5%–98.6% accuracy even without explicit keywords.
- **📈 Quiz Score Regressor:** Predicts post-session assessment scores (0–100%) with 7.65% MAE using zero-leakage student grouping.
- **📊 Interactive Streamlit & Vercel Dashboards:**
  - **Student Performance Analyzer:** Test real-time inference on sample dialogues, paste custom transcripts, or upload files with live feature attributions.
  - **Topic-Wise Performance:** Curriculum-level mastery aggregates, confusion rates, and topic difficulty rankings.
  - **Model Architecture & Viva Guide:** Multi-model benchmarks, 6 ablation experiments, explainability tabs, and defense Q&A.
- **⚡ Deterministic & Explainable:** Sub-millisecond inference per session with transparent probability distributions and explainable linear coefficients.
- **🛡️ 25 Automated Unit & Integration Tests:** Powered by `pytest` ensuring 100% end-to-end pipeline reliability.

---

## 🗂️ Repository Structure

```text
EduLens/
├── .gitignore                             # Git ignore rules (caches, venvs, env files)
├── pytest.ini                             # Pytest configuration (pythonpath = .)
├── requirements.txt                       # Production & prototype dependencies
├── PROJECT_AUDIT.md                       # Comprehensive codebase & architecture audit
├── README.md                              # Main onboarding guide (this file)
├── api/                                   # Vercel Serverless API endpoints
│   └── index.py                           # FastAPI application (/api/predict, /api/explainability, etc.)
├── data/
│   ├── raw/
│   │   ├── edulens_chats_dataset.json     # Primary supervised dataset (430 sessions)
│   │   ├── edulens_chats_set1 (1).json    # Supplementary dataset (420 sessions)
│   │   ├── edulens_chats_set2 (1).json    # Supplementary dataset (430 sessions)
│   │   ├── edulens_chats_set3 (1).json    # Supplementary dataset (440 sessions)
│   │   ├── education_dialogue/            # 47k Gemini dialogues (unlabeled)
│   │   └── mathdial/                      # 2.8k human math tutoring dialogues
│   └── processed/                         # Validated, leakage-free cleaned datasets
├── docs/                                  # Four Permanent Technical Records
│   ├── DSP.md                             # Data Science Pipeline specification
│   ├── NLP.md                             # Natural Language Processing & vectorization
│   ├── ML.md                              # Machine Learning benchmarks & ablation
│   └── DATASET.md                         # Dataset schemas & group splitting rules
├── ml_prototype/                          # Core Machine Learning Package
│   ├── __init__.py                        # Package root
│   ├── data_loader.py                     # Schema normalization & dataset loader
│   ├── features.py                        # Conversational signals, TF-IDF, & ablation pipelines
│   ├── explainability.py                  # Model coefficients & instance attribution
│   ├── train_model.py                     # Group-aware training, CV & artifact generation
│   ├── predict.py                         # Multi-model inference engine with explainability
│   └── dashboard.py                       # 3-page interactive Streamlit dashboard
├── models/                                # Versioned Model Artifacts
│   ├── trained_model.joblib               # Primary backward-compatible pipeline artifact
│   ├── understanding/                     # Understanding classification models
│   │   ├── logistic_regression/           # Calibrated Logistic Regression model & metadata
│   │   ├── linear_svm/                    # Calibrated Linear SVM model & metadata
│   │   └── naive_bayes/                   # Multinomial Naive Bayes model & metadata
│   └── quiz_score/                        # Quiz score regression model (Random Forest / Ridge)
├── reports/                               # Generated JSON & Markdown Reports
│   ├── feature_ablation.json              # 6-configuration feature ablation report
│   ├── model_comparison.json              # Cross-model evaluation benchmark
│   ├── model_evaluation.json              # Primary Logistic Regression evaluation report
│   ├── model_explainability.json          # Global explainability weights and importances
│   ├── prediction_walkthrough.md          # 7-stage dialogue prediction trace
│   └── quiz_regressor_evaluation.json     # Group-aware regression evaluation
├── scripts/                               # Reproducible Execution Scripts
│   ├── prepare_data.py                    # Dataset validation and cleaning
│   ├── run_feature_ablation.py            # Systematic ablation experiment suite
│   └── walkthrough_prediction.py          # Prediction trace generator
└── tests/
    ├── __init__.py
    ├── test_api.py                        # FastAPI endpoint test suite
    └── test_prototype.py                  # Data loader, pipeline, model & explainability tests
```

---

## 📊 Machine Learning Architecture

### Prediction Target
- **`final_understanding`:** 3-class categorical label representing demonstrated conceptual mastery:
  - **`understood`** (44.7%): Student articulated concepts correctly, resolved questions, and confirmed clarity.
  - **`partial`** (36.3%): Student understood core ideas but required repeated clarifications or exhibited hesitation.
  - **`confused`** (19.1%): Student expressed persistent difficulty, repeated doubts, or failed to apply concepts.

### Feature Pipeline (108 Features)
1. **Dense Conversational Signals (8 Features, Scaled via StandardScaler / MinMaxScaler):**
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

Evaluated on held-out test split (73 sessions from 6 unseen students, 0 overlap) with 5-Fold Stratified Group Cross-Validation on `student_id`:

| Metric | Dummy Baseline (Majority) | Logistic Regression (Balanced) | Multinomial Naive Bayes | Linear SVM (Calibrated) |
|---|---|---|---|---|
| **Test Accuracy** | 53.42% | **95.89%** | 80.82% | **100.00%** |
| **Macro Precision** | 0.1781 | **0.9380** | 0.7569 | **1.0000** |
| **Macro Recall** | 0.3333 | **0.9429** | 0.7167 | **1.0000** |
| **Macro F1-Score** | 0.2321 | **0.9399** | 0.7299 | **1.0000** |
| **Weighted F1-Score** | 0.3718 | **0.9591** | 0.7966 | **1.0000** |
| **5-Fold CV Accuracy** | — | **93.78% (± 3.23%)** | 84.19% (± 10.20%) | **97.77% (± 1.82%)** |
| **5-Fold CV Macro F1** | — | **92.95% (± 3.14%)** | 81.34% (± 9.92%) | **97.44% (± 2.10%)** |

### Secondary Model: Quiz Score Regressor (Group-Aware Split)
- **Dataset:** 1,720 sessions across 40 students (`GroupShuffleSplit` by `student_id`, 1,394 train / 326 test, 0 overlap).
- **Random Forest Regressor:** MAE = **7.65%**, RMSE = **9.36%**, $R^2 = 0.7624$
- **Ridge Regressor (L2):** MAE = **7.76%**, RMSE = **9.68%**, $R^2 = 0.7460$

---

## 🌐 Deploy to Vercel

EduLens is fully configured for **serverless deployment on Vercel** out-of-the-box!

### How It Works on Vercel
- **Serverless API (`api/index.py`):** Fast, stateless Python functions running on Vercel Serverless Functions. Handles real-time ML inference (`/api/predict`), curriculum analytics (`/api/topics`), benchmark samples (`/api/samples`), and model health (`/api/health`).
- **Interactive Web App (`public/index.html`):** Modern, lightweight, responsive client dashboard powered by Tailwind CSS, Chart.js, and Lucide icons.
- **Fast & Lightweight:** Unnecessary training datasets are excluded via `.vercelignore`, making deployment zip bundles ultra-compact (< 2MB).

### Deploy in 2 Steps:
1. **Import from GitHub:**
   - Go to [vercel.com](https://vercel.com/) and click **"Add New Project"**.
   - Select your repository **`chiragchauhan07/EduLens`**.
   - Vercel automatically detects `vercel.json` and `requirements.txt`.
   - Click **Deploy**.

2. **Or Deploy via Vercel CLI:**
   ```powershell
   npm install -g vercel
   vercel
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
