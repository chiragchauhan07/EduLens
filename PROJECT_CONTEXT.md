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

## 3. Scope Simplification for Tomorrow's Showcase

To ensure maximum stability, instant startup, and zero broken dependencies, the following enterprise and experimental modules have been **decoupled from the active runtime** and archived:

- **Archived Features**:
  - Model Context Protocol (MCP) server, SSE transports, OAuth/token auth (`archive/mcp_server/`)
  - Continuous external ChatGPT / Claude live stream ingestion
  - Dynamic quiz generation and quiz score regression
  - Multi-tenant student database management
  - Complex multi-page analytics dashboards

- **Retained & Hardened Core**:
  - Full NLP and conversational feature extraction pipeline (108 dimensions)
  - Pre-trained Multinomial Logistic Regression classifier (95.89% accuracy)
  - Expressed linguistic confidence estimator
  - Multi-schema JSON conversation parser
  - Clean FastAPI backend (`backend/main.py`, `backend/routes/analysis.py`)
  - Stitch-designed minimalist frontend (`frontend/index.html`, `frontend/assets/app.js`)
  - 57-test automated verification suite

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
├── backend/
│   ├── main.py                     # Application entrypoint & static mounting
│   ├── routes/
│   │   └── analysis.py             # Active API routes
│   ├── services/
│   │   ├── conversation_parser.py  # Multi-schema JSON parser
│   │   ├── preprocessing.py        # DataFrame formatter
│   │   ├── feature_extraction.py   # Conversational signals & confidence
│   │   └── prediction.py           # Model singleton & analysis engine
│   └── models/
│       ├── understanding_classifier.joblib
│       └── metadata.json
├── frontend/
│   ├── index.html                  # Apple/Google aesthetic UI
│   └── assets/
│       ├── app.js                  # Frontend interactions
│       └── styles.css              # Custom styling
├── samples/                        # 8 Test conversations (understood, confused, etc.)
├── ml_prototype/                   # Historical data loader & training scripts
├── archive/                        # Decoupled MCP server & older tests
└── tests/                          # 57 Passing Pytest unit and integration tests
```
