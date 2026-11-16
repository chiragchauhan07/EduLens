# EduLens

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![MCP](https://img.shields.io/badge/MCP-Official%202.x-purple.svg)](docs/MCP.md)
[![Tests](https://img.shields.io/badge/Tests-89%20Passing-brightgreen.svg)](#running-tests)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-MCP%20Server%20Verified-success.svg)](#run-model-context-protocol-mcp-server)

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

- **🎯 Four Core Performance Dimensions:**
  1. *Topic Understanding:* 3-class ML classification (`Understood`, `Partial`, `Confused`) + transparent 0–100 score mapping.
  2. *Actual Quiz Performance:* Deterministic grading comparing student answers vs answer key (distinct from conversational regression).
  3. *Learning Confidence:* Expressed linguistic confidence estimator extracting 7 distinct conversational discourse markers.
  4. *Overall Topic Performance:* Composite 40/40/20 scoring engine with graceful handling of missing quiz submissions and discrepancy detection.
- **🔄 Unified Inference Pipeline:** `EduLensUnifiedPipeline` orchestrating DSP validation, NLP parsing, ML inference, and multi-dimensional scoring into one production JSON schema contract.
- **🔍 Multi-Schema Session Ingestion:** Accepts conversation logs across standard formats (`turns`, `messages`, raw dialogue, or multi-topic JSON payloads).
- **🧪 Leakage-Free Feature Engineering:** 108 features combining 8 dense pedagogical & structural signals with 100 TF-IDF lexical n-grams.
- **🎯 High-Precision Understanding Classifier:** Categorizes comprehension into `understood`, `partial`, and `confused` with **95.89% accuracy** (Logistic Regression) and **100.0% accuracy** (Linear SVM) under zero-leakage group splitting.
- **💡 Built-in Explainability Engine:** Decomposes predictions into mathematical feature contributions, identifying exact dialogue drivers for each class.
- **🔬 6-Configuration Feature Ablation:** Proves dialogue structure alone carries 75.3% accuracy, and the model preserves 94.5%–98.6% accuracy even without explicit keywords and shortcut vocabulary.
- **📈 Quiz Score Regressor:** Predicts post-session assessment scores (0–100%) with 7.65% MAE using zero-leakage student grouping.
- **📊 Interactive Streamlit & Vercel Dashboards:**
  - **Student Performance Analyzer:** Test real-time inference, optional quiz submission evaluation, discrepancy alerts, and feature attributions.
  - **Topic-Wise Performance:** Curriculum-level mastery aggregates, confusion rates, and topic difficulty rankings.
  - **Model Architecture & Viva Guide:** Multi-model benchmarks, 6 ablation experiments, explainability tabs, and defense Q&A.
- **🛡️ 78 Automated Unit & Integration Tests:** Powered by `pytest` verifying ML, API, security, multi-user isolation, Streamable HTTP, SSE, and Stdio Model Context Protocol integration with a 100% test suite pass rate.
- **🔌 Model Context Protocol (MCP) Server:** Standard MCP 2.x server with 6 core tools for session ingestion, real-time analytics, longitudinal mastery, and quiz submission across Streamable HTTP (`/mcp`), SSE, and Stdio transports; verified with Claude Desktop and MCP Inspector.
- **💾 Thread-Safe Persistent SQLite Storage:** Thread-safe WAL-mode SQLite storage separating audit-proof raw conversation transcripts (`raw_sessions`) from derived learning analytics (`analytics_results`) and quiz evaluations (`quiz_submissions`).
- **🔐 Multi-User Isolation & Context Propagation:** Cryptographic bearer token management mapping requests to isolated student IDs with automatic deduplication via SHA-256 session hashing.

---

## 🗂️ Repository Structure

```text
EduLens/
├── .env.example                           # Template environment configuration
├── .gitignore                             # Git ignore rules (caches, venvs, env files, DBs)
├── pytest.ini                             # Pytest configuration (pythonpath = .)
├── requirements.txt                       # Production, ML & MCP dependencies
├── PROJECT_AUDIT.md                       # Comprehensive codebase & architecture audit
├── README.md                              # Main onboarding guide (this file)
├── api/                                   # Vercel Serverless API endpoints
│   └── index.py                           # FastAPI application (/api/pipeline, /api/predict, etc.)
├── backend/                               # Backend Services & Protocols
│   └── mcp_server/                        # Official Model Context Protocol (MCP) Server
│       ├── auth/                          # TokenManager & user context isolation
│       ├── schemas/                       # Pydantic schemas for 6 tools & resources
│       ├── services/                      # IngestionService & SQLiteStorageService (WAL)
│       ├── tools/                         # Modular tool handlers (ingestion, analytics, quiz)
│       ├── config.py                      # Server configuration & environment variables
│       └── server.py                      # MCP entry point (SSE, Streamable, Stdio)
├── data/
│   ├── raw/                               # Historical conversation datasets (1,700+ sessions)
│   ├── processed/                         # Validated, leakage-free cleaned datasets
│   └── storage/                           # Persistent storage (data/storage/edulens.db)
├── docs/                                  # Technical Specifications & Architecture Records
│   ├── DATASET.md                         # Dataset schemas & group splitting rules
│   ├── DSP.md                             # Data Science Pipeline specification
│   ├── ML.md                              # Machine Learning benchmarks & ablation
│   ├── MCP.md                             # Model Context Protocol full architecture & guide
│   └── NLP.md                             # Natural Language Processing & vectorization
├── ml_prototype/                          # Core Machine Learning Package
│   ├── confidence.py                      # Expressed linguistic confidence estimator
│   ├── dashboard.py                       # 3-page interactive Streamlit dashboard
│   ├── data_loader.py                     # Schema normalization & dataset loader
│   ├── dsp.py                             # Modular DSP pipeline (validation, speaker dynamics)
│   ├── explainability.py                  # Model coefficients & instance attribution
│   ├── features.py                        # Conversational signals, TF-IDF, & ablation pipelines
│   ├── nlp.py                             # Modular NLP pipeline (parsing, tokenization, 108-dim)
│   ├── predict.py                         # Multi-model inference engine with explainability
│   ├── quiz_evaluator.py                  # Deterministic quiz grading engine
│   ├── scoring_engine.py                  # Composite scoring (40/40/20 formula & discrepancy)
│   ├── train_model.py                     # Group-aware training, CV & artifact generation
│   └── unified_pipeline.py                # Unified end-to-end pipeline (EduLensUnifiedPipeline)
├── models/                                # Versioned Model Artifacts
│   ├── quiz_regressor/                    # Quiz score regression models (Random Forest / Ridge)
│   ├── understanding/                     # Understanding classification models (LR, SVM, NB)
│   └── trained_model.joblib               # Backward-compatible pipeline artifact
├── reports/                               # Generated JSON & Markdown Reports
├── scripts/                               # Reproducible Execution Scripts
└── tests/
    ├── test_api.py                        # FastAPI endpoint test suite (11 tests)
    ├── test_comprehensive_suite.py        # 12 acceptance cases & modular tests (16 tests)
    ├── test_mcp_e2e.py                    # Live ClientSession MCP subprocess E2E test
    ├── test_mcp_server.py                 # MCP server suite: all 6 tools, storage & auth (24 tests)
    └── test_prototype.py                  # Pipeline, model & explainability tests (14 tests)
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

### 🚀 Launch the Full EduLens Application (FastAPI + 8-Page Dashboard)
The primary entrypoint serves both the REST API and the complete 8-page Google Stitch analytics web dashboard:
```powershell
uvicorn api.index:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser to access the live application:
- **1. Overview Dashboard:** Longitudinal KPIs, understanding distribution donut, progression trend charts, areas of struggle.
- **2. Learning Sessions:** Filterable, paginated session repository with composite scores and quick-inspect slide-over drawer.
- **3. Session Analytics Detail:** 4-dimension performance breakdown, local feature attributions, pedagogical recommendations, linked quizzes.
- **4. Topics & Curriculum:** Studied vs unstudied curriculum topics, mastery progression, and curriculum benchmarks.
- **5. Performance Analytics:** Longitudinal score distribution, multi-agent comparison, and confusion diagnosis.
- **6. Quiz Evaluations:** Deterministic quiz evaluation, question-by-question breakdown, and composite score recalculation.
- **7. Agent Integration & Live Inference:** Multi-algorithm inference playground, local feature drivers, MCP connection guides.
- **8. Settings & Student Isolation:** Local identity switcher (Dev mode) and Bearer token security diagnostics (Production mode).

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

### Run Model Context Protocol (MCP) Server
EduLens includes a standard MCP 2.x server allowing Claude Desktop, ChatGPT, and MCP Inspector to interact directly with the learning analytics engine:

```powershell
# Run with Streamable HTTP for production deployments & ChatGPT (Recommended, port 8001, endpoint /mcp)
python -m backend.mcp_server.server --transport streamable --port 8001

# Run with Server-Sent Events (SSE) for legacy remote clients (port 8001, endpoint /sse)
python -m backend.mcp_server.server --transport sse --port 8001

# Run with Stdio for local Claude Desktop
python -m backend.mcp_server.server --transport stdio
```

See [docs/MCP.md](docs/MCP.md) for full Claude Desktop, ChatGPT, and MCP Inspector configuration guides.

### Run Automated Tests
Verify code integrity across the full test suite (89 tests):
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
- [x] **Phase 2: Live Ingestion & MCP Integration** — Model Context Protocol server (official MCP 2.x, SSE/Stdio transports, 6 production tools, WAL SQLite, 66 passing tests).
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
