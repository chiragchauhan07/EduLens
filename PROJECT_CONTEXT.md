# EduLens — Project Context & System Architecture

> **Document Status:** Active & Aligned with Repository State  
> **Target Audience:** Engineering Team, Data Scientists, Contributors, Reviewers  
> **Last Updated:** November 2026  

---

## 1. Executive Summary & Core Philosophy

**EduLens** is an open, production-oriented student learning intelligence and performance analytics platform.

### What EduLens Is:
- A post-learning assessment and intelligence layer built around the student's natural learning workflow.
- A platform that ingests learning dialogues from external LLM clients (such as ChatGPT, Claude, and Gemini) via standard protocols like Model Context Protocol (MCP).
- A system that extracts quantifiable, interpretable linguistic and pedagogical signals from conversation transcripts to predict demonstrated comprehension, detect learning struggles, diagnose confusion, and evaluate curriculum mastery across topics.

### What EduLens Is NOT:
- **EduLens is NOT an AI tutor.** It does not attempt to replace classroom instructors or serve as another conversational chatbot answering homework questions.
- Students continue using whatever AI assistant or tutoring tool they prefer. EduLens provides the **measurement, diagnostics, and long-term learning progression tracking** that external LLM interfaces inherently lack.

---

## 2. End-to-End System Architecture & Data Flow

EduLens operates on an incremental pipeline that bridges conversational interaction with longitudinal learning analytics:

```
[Student Learns via External LLM] (ChatGPT / Claude / Local Client)
                 │
                 ▼
[MCP Session Export / Direct Ingestion]
                 │  (Canonical JSON Session Payload)
                 ▼
[Data Cleaning & Boundary Validation]
                 │  (Isolate student utterances, strip secrets/PII)
                 ▼
[Deterministic NLP & Feature Engineering]
                 │  (108 features: 8 conversational signals + 100 TF-IDF n-grams)
                 ▼
[Machine Learning Inference Pipeline]
                 │  (Multinomial Logistic Regression: Understood / Partial / Confused)
                 ▼
[Diagnostic Feedback & Targeted Quiz Generation]
                 │  (Gemini API generates focused questions on diagnosed weak topics)
                 ▼
[Student Learning State & Longitudinal Profiling]
                 │  (Track mastery, confusion trends, and learning curves over time)
                 ▼
[Interactive Analytics Dashboard & Educator Reports]
```

---

## 3. Core Architecture Principles

1. **Deterministic Processing Where Possible:**  
   JSON parsing, boundary validation, session storage, feature extraction, scaling, and statistical aggregation are deterministic and reproducible. We avoid routing data cleaning or basic classification through nondeterministic LLM calls.

2. **Explainable ML Over Black-Box Complexity:**  
   Models used for educational evaluation must be transparent and explainable. For student defense, viva examinations, and instructor trust, decision drivers (e.g., specific confusion markers, turn counts, vocabulary keywords) can be directly traced back to feature coefficients and calibrated class probabilities.

3. **Strategic Use of Generative AI:**  
   Generative LLMs (e.g., Google Gemini) are reserved strictly for high-value generative tasks—specifically synthesizing dynamic, targeted 10–15 question quizzes tailored to a student's diagnosed confusion areas—not for deterministic feature calculation.

4. **Privacy-by-Design & Data Minimalism:**  
   Treat student conversations as strictly private. EduLens ingests *only* sessions explicitly submitted by the user. Raw logs and processed feature matrices are separated, credentials and tokens are secured via environment variables, and data deletion requests are honored.

5. **Decoupled & Source-Agnostic Design:**  
   Whether conversations arrive from Claude Desktop via stdio MCP, ChatGPT web via Streamable HTTP, or batch JSON files, data is normalized into a unified internal schema (`chat_id`, `student_id`, `subject`, `topic`, `student_messages`, `turns`).

---

## 4. Current Implementation State

The repository contains the fully functional, verified **Machine Learning Prototype** alongside evaluation reports and datasets:

| Component | Status | Location | Description |
|---|---|---|---|
| **Data Loader** | ✅ Implemented | [`ml_prototype/data_loader.py`](ml_prototype/data_loader.py) | Universal parser and schema normalizer for multi-turn sessions. |
| **Feature Extractor** | ✅ Implemented | [`ml_prototype/features.py`](ml_prototype/features.py) | Leakage-free pipeline generating 108 conversational & lexical features. |
| **Model Training** | ✅ Implemented | [`ml_prototype/train_model.py`](ml_prototype/train_model.py) | Reproducible training script with 5-fold CV, baseline comparison, and artifact export. |
| **Inference Engine** | ✅ Implemented | [`ml_prototype/predict.py`](ml_prototype/predict.py) | Real-time classification engine delivering confidence scores and pedagogical feedback. |
| **Analytics Dashboard** | ✅ Implemented | [`ml_prototype/dashboard.py`](ml_prototype/dashboard.py) | 3-page interactive Streamlit app (Student Analyzer, Topic Summary, Viva Guide). |
| **Test Suite** | ✅ Implemented | [`tests/test_prototype.py`](tests/test_prototype.py) | Automated test suite verifying loaders, feature extraction, and inference. |
| **Datasets** | ✅ Available | [`data/raw/`](data/raw/) | 430 supervised sessions (`edulens_chats_dataset.json`), supplementary sets, and research dialogue corpora. |
| **Model Artifact** | ✅ Serialized | [`models/trained_model.joblib`](models/trained_model.joblib) | Production-ready Scikit-learn Pipeline artifact. |
| **Evaluation Reports** | ✅ Documented | [`reports/`](reports/) | Rigorous audit reports: `dataset_inspection.md` and `model_evaluation.json`. |

---

## 5. Repository Component Boundaries

```
EduLens/
├── data/                               # Dataset storage
│   └── raw/                            # Supervised datasets and research corpora
├── ml_prototype/                       # Core ML package
│   ├── __init__.py                     # Package initialization
│   ├── data_loader.py                  # Ingestion & normalization logic
│   ├── features.py                     # Feature engineering & NLP transformers
│   ├── train_model.py                  # Training pipeline & evaluation
│   ├── predict.py                      # Real-time inference wrapper
│   └── dashboard.py                    # Streamlit visual analytics app
├── models/                             # Serialized model artifacts
│   └── trained_model.joblib            # Trained Scikit-learn pipeline
├── reports/                            # Experimental findings & validation
│   ├── dataset_inspection.md           # Dataset schema & quality report
│   └── model_evaluation.json           # Model performance benchmarks
├── tests/                              # Automated test suite
│   ├── __init__.py
│   └── test_prototype.py               # Unit & integration tests
├── EduLens Project Specification.pdf  # Foundational specification document
├── generate_dataset.py                 # Pedagogical synthetic dataset generator
├── PROJECT_CONTEXT.md                  # This architecture reference document
├── prototype_context.md                # ML implementation & viva defense guide
├── README.md                           # Main repository onboarding & quickstart
├── requirements.txt                    # Pinned Python package dependencies
└── pytest.ini                          # Test configuration
```

---

## 6. Phased Engineering Roadmap

- **Phase 1: ML Prototype & Feasibility (COMPLETED)**  
  Curated dataset, built leakage-free feature extraction, trained 96.5% accurate interpretable model, and deployed 3-page Streamlit dashboard.

- **Phase 2: Automated Ingestion & MCP Integration (NEXT)**  
  Implement the standard Model Context Protocol (MCP) server supporting `stdio` (for Claude Desktop) and `streamable-http` (for ChatGPT / web clients) to ingest conversations live.

- **Phase 3: Targeted Quiz Generation & Gemini Orchestration**  
  Integrate Google Gemini API to analyze diagnosed confusion points from Phase 1 and synthesize adaptive multiple-choice quizzes to validate comprehension empirically.

- **Phase 4: Longitudinal Modeling & Student Knowledge Tracing**  
  Persist historical sessions per `student_id` in a relational database (PostgreSQL) to construct Bayesian Knowledge Tracing (BKT) or longitudinal mastery curves across academic semesters.

- **Phase 5: Full-Stack Web Platform & Educator Portal**  
  Transition from Streamlit to a production Next.js frontend with FastAPI backend, role-based access control (students vs. instructors), and PDF cohort performance reporting.

---

## 7. Guidelines for Collaborators

When contributing to EduLens:
1. **Maintain Determinism:** Keep feature engineering reproducible; always fix seeds (`random_state=42`) in stochastic operations.
2. **Prevent Data Leakage:** Ensure transformers and scalers are fitted exclusively on training splits inside Scikit-learn `Pipeline` objects.
3. **Verify Before Pushing:** Always run `pytest` and `python ml_prototype/predict.py` locally before committing.
4. **Document Decisions:** If adjusting features or model hyperparameters, update both `reports/model_evaluation.json` and `prototype_context.md`.
