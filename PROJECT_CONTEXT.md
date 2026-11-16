# EduLens — Project Context & System Architecture

> **Document Status:** Active & Aligned with Final ML Practical Showcase  
> **Target Audience:** Examiners, Faculty Reviewers, Data Scientists, Contributors  
> **Version:** 1.0.0 (EduLens Production Edition)  

---

## 1. Executive Summary & Purpose

**EduLens** is a focused, lightweight Machine Learning application designed for college ML practical demonstration.

It simplifies the broader EduLens vision into a single, high-reliability core workflow:
**Analyze an uploaded student learning conversation in JSON format, extract NLP and conversational features, predict student topic understanding, and estimate expressed linguistic confidence.**

### Key Principles:
1. **Independent Evaluation Layer**: EduLens is not an AI tutor. It does not generate conversational responses. It analyzes dialogues between a student and an external tutor/AI.
2. **Local, Lightweight Inference**: Zero external AI API calls or cloud dependencies required during inference.
3. **High Explainability**: Combines 100 domain TF-IDF vocabulary n-grams with 8 dense pedagogical conversational signals into a 108-dimensional feature vector.
4. **Methodological Honesty**: Clearly separates supervised ML classification (understanding) from rule-calibrated discourse analysis (expressed confidence). Does not claim to measure psychological conviction.

---

## 2. Active Application Scope & Architecture

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

## 3. Production Architecture & Simplification

To ensure maximum stability, instant startup, zero broken dependencies, and seamless Vercel serverless deployment, the repository was streamlined into a dedicated single-purpose system:

- **Pruned Unused Features**:
  - Model Context Protocol (MCP) server, SSE transports, OAuth/token authentication (decoupled)
  - Continuous external ChatGPT / Claude live stream connectors
  - Dynamic quiz generation and 3.3MB quiz score regressor model
  - Redundant frontend mirrors (`public/`) and legacy prototype models
  - Untracked SQLite databases

- **Retained & Production-Hardened Core**:
  - Full NLP and conversational feature extraction pipeline (108 dimensions)
  - Pre-trained Multinomial Logistic Regression classifier (95.89% accuracy)
  - Expressed linguistic confidence estimator
  - Multi-schema JSON conversation parser
  - Clean FastAPI backend (`backend/main.py`, `backend/routes/analysis.py`)
  - Modern minimalist frontend (`frontend/index.html`, `frontend/assets/app.js`, `frontend/assets/styles.css`)
  - Vercel Serverless configuration (`api/index.py`, `vercel.json`, `.vercelignore`)
  - 57-test automated verification suite (`pytest tests/ -v`)

---

## 4. Deployed ML Pipeline Specifications

### A. Preprocessing & Normalization
- Input: Multi-turn educational dialogue transcript in JSON.
- Extraction: Isolates student utterances (`student_messages`), concatenates into `full_text`.
- Cleaning: Normalizes unicode quotation marks, collapses whitespace, preserves pedagogical punctuation (`?`).

### B. Feature Space (108 Total Dimensions)
1. **TF-IDF Vocabulary Bank (100 Features)**:
   - Scikit-Learn `TfidfVectorizer(max_features=100, ngram_range=(1, 2), min_df=2, stop_words='english')`.
   - Models domain terminology, causal conjunctions (*because*, *therefore*, *which means*), and topic keywords.
2. **Conversational Signal Matrix (8 Features)**:
   - Scikit-Learn `StandardScaler()` applied to:
     - `num_turns`, `total_words`, `avg_turn_length`, `question_count`
     - `confusion_markers`, `clarification_markers`, `understanding_markers`, `code_indicators`

### C. Understanding Classifier
- **Model**: Scikit-Learn `Pipeline` containing `ColumnTransformer` + `LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)`.
- **Classes**:
  1. `Understood` (Index 2)
  2. `Partially Understood` (Index 1)
  3. `Confused` (Index 0)
- **Metrics**: 95.89% Test Accuracy, 0.9399 Macro F1-score, 93.78% 5-fold CV accuracy.

### D. Expressed Confidence Estimator
- **Model Type**: Transparent rule-calibrated linguistic indicator (NOT an ML model).
- **Markers**: Uncertainty hedges, certainty assertions, hesitation fillers, self-corrections, assertion-hedge ratio.
- **Output**: High Confidence (> 68%), Moderate Confidence (40%–68%), Low Confidence (< 40%).

---

## 5. Active API Routes

| HTTP Verb | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the single-page application frontend |
| `GET` | `/api/health` | Health check verifying model registry status |
| `GET` | `/api/model-info` | Returns model architecture, metrics, classes, and pipeline specs |
| `GET` | `/api/sample-conversation` | Returns pre-loaded Physics Friction dialogue for demo |
| `POST` | `/api/analyze` | Accepts multipart `.json` file upload or direct JSON body; runs inference |

---

## 6. Directory Layout

```text
EduLens/
├── api/
│   └── index.py                    # Vercel serverless ASGI entrypoint
├── backend/
│   ├── main.py                     # Application entrypoint & static mounting
│   ├── routes/
│   │   └── analysis.py             # Active API routes (/health, /model-info, /analyze)
│   ├── services/
│   │   ├── conversation_parser.py  # Multi-schema JSON parser & validator
│   │   ├── preprocessing.py        # DataFrame formatter & utterance cleaner
│   │   ├── feature_extraction.py   # Conversational signals & confidence estimator
│   │   └── prediction.py           # Model singleton & analysis engine
│   └── models/
│       ├── understanding_classifier.joblib  # Active 108-dim trained pipeline (9.9 KB)
│       └── metadata.json           # Model specifications & CV metrics
├── frontend/
│   ├── index.html                  # Responsive minimalist web interface
│   └── assets/
│       ├── app.js                  # Frontend interactions & signal rendering
│       └── styles.css              # Custom styling & animation tokens
├── samples/                        # 8 Standard test conversations
├── ml_prototype/                   # Reproducible data loader, training, and evaluation scripts
├── data/                           # Labeled training datasets & reference corpora
├── tests/                          # 57 Passing Pytest unit and integration tests
├── requirements.txt                # Lean production dependencies (8 packages)
├── requirements-dev.txt            # Development & testing dependencies
├── vercel.json                     # Edge rewrite configuration
├── .vercelignore                   # Serverless bundle exclusions
└── README.md                       # Public documentation
```

---

## 7. Current Limitations & Scope

1. **Dialogue Evidence Only**: Evaluates explicit textual expressions in the provided dialogue. Does not measure latent cognitive ability or long-term retention.
2. **Vocabulary Domain**: The 100-dimensional TF-IDF lexical bank focuses on STEM and Computer Science topics. Non-STEM conversations rely on conversational and structural markers.
3. **Linguistic Confidence vs. Truth**: Expressed confidence evaluates conversational certainty and hedging, not factual correctness. A student can sound confident while asserting a misconception.
