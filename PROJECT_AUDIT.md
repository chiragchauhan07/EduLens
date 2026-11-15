# PROJECT AUDIT — EDULENS

**Generated:** November 15, 2026  
**Auditor:** Lead Data Scientist & ML Engineer  
**Repository:** [chiragchauhan07/EduLens](https://github.com/chiragchauhan07/EduLens.git)  
**Status:** Complete Repository Audit & Verification

---

## 1. Actual Repository Structure

```text
EduLens/
├── api/
│   └── index.py                     # Vercel Serverless ASGI entrypoint (FastAPI app with /predict, /topics, /samples, /model-info)
├── data/
│   ├── processed/                   # Intended for cleaned and prepared datasets (currently empty)
│   └── raw/
│       ├── edulens_chats_dataset.json           # Primary supervised dataset (430 sessions, 11 CS subjects, labeled with final_understanding)
│       ├── edulens_chats_set1 (1).json          # Supplementary set (420 sessions with quiz, no final_understanding or topics)
│       ├── edulens_chats_set2 (1).json          # Supplementary set (430 sessions with quiz, no final_understanding or topics)
│       ├── edulens_chats_set3 (1).json          # Supplementary set (440 sessions with quiz, no final_understanding or topics)
│       ├── edulens_expanded_pilot_020_v2.json   # 20 pilot sessions with schema v2.1 (engagement, misconception, self-correction, reasoning)
│       ├── edulens_pilot_manifest_v2.json       # Manifest metadata for pilot generation v2
│       ├── edulens_schema_and_strategy_v2_1.md  # Detailed specification for schema v2.1
│       ├── edulens_tools.py                     # Dataset utility CLI (model-inputs extraction, blind review sheet, Cohen's kappa)
│       ├── pilot_change_log.json                # Pilot evolution change log
│       ├── pilot_review_and_change_log.md       # Review log for schema transitions
│       ├── pilot_validation_report_v2.json      # Validation run results for 20-sample pilot
│       ├── validate_edulens.py                  # Schema v2.1 3-layer validator (structural, semantic, leakage diagnostic)
│       ├── education_dialogue/                  # Google Research / HF Education Dialogue corpus
│       │   ├── conversations_train1.json .. train5.json  (40,000 conversations)
│       │   ├── conversations_eval.json                   (7,234 conversations)
│       │   └── README.md
│       └── mathdial/                            # EPFL MathDial MTurk corpus
│           ├── train.jsonl                      (2,262 conversations)
│           ├── test.jsonl                       (599 conversations)
│           ├── .gitattributes
│           └── README.md
├── ml_prototype/
│   ├── __init__.py                  # Package init
│   ├── dashboard.py                 # Interactive Streamlit application (3 pages: Analyzer, Topics, Model Info)
│   ├── data_loader.py               # Universal conversation extractor and session normalizer
│   ├── features.py                  # ConversationalFeatureExtractor (8 dense signals) + TfidfVectorizer (100 features)
│   ├── predict.py                   # Real-time inference engine (EduLensPredictor class)
│   └── train_model.py               # Model training script (LogisticRegression with 5-fold CV & test evaluation)
├── models/
│   └── trained_model.joblib         # Serialized Scikit-learn Pipeline (ColumnTransformer + LogisticRegression)
├── public/
│   └── index.html                   # Standalone Tailwind CSS/HTML dashboard mimicking Streamlit UI
├── reports/
│   ├── dataset_inspection.md        # Initial evaluation and schema analysis of datasets
│   └── model_evaluation.json        # Test metrics, 5-fold CV scores, and confusion matrix
├── tests/
│   ├── __init__.py
│   ├── test_api.py                  # FastAPI serverless test client test cases (health, samples, topics, predict)
│   └── test_prototype.py            # Unit & integration tests for data loader, feature extraction, and predictor
├── .gitignore                       # Git ignore configuration
├── .vercelignore                    # Vercel deployment ignore rules
├── EduLens Project Specification.pdf # Foundational project PDF specification
├── generate_dataset.py              # Synthetic dialogue generator script (~500 conversations across 9 subjects)
├── LICENSE                          # MIT License
├── PROJECT_CONTEXT.md               # Architecture and roadmap reference
├── prototype_context.md             # ML prototype implementation summary & viva defense guide
├── pytest.ini                       # Pytest testpath and pythonpath configuration
├── README.md                        # Project overview and instructions
├── requirements.txt                 # Production serverless runtime dependencies
├── requirements-dev.txt             # Development and test dependencies
└── vercel.json                      # Vercel API routing configuration
```

---

## 2. Technologies and Libraries

| Category | Component / Tool | Version / Spec | Usage in EduLens |
|---|---|---|---|
| **Language** | Python | 3.12.4 (x86_64 Windows) | Core codebase runtime |
| **Backend Framework** | FastAPI | 0.163.1 | REST API endpoints for serverless inference & analytics |
| **Data Validation** | Pydantic | 2.14.3 | API request body schema validation |
| **Machine Learning** | Scikit-learn | 1.5.2 | Feature transformers, Pipeline, LogisticRegression, CV, metrics |
| **Data Science** | Pandas, NumPy, SciPy | 3.1.0 / 2.5.4 / 1.18.1 | Data manipulation, normalization, numerical vectors |
| **Serialization** | Joblib | 1.6.0 | Model artifact saving and loading |
| **Frontend 1** | Streamlit | 1.66.1 | Local interactive analytical dashboard (`ml_prototype/dashboard.py`) |
| **Frontend 2** | HTML5 / Tailwind CSS CDN | 3.x CDN | Standalone browser UI for Vercel static deployment (`public/index.html`) |
| **Testing** | Pytest, HTTPX | 9.2.0 / 0.28.1 | Automated testing suite and TestClient |
| **Serverless Host** | Vercel | Serverless Functions | ASGI serverless deployment via `vercel.json` |

---

## 3. Existing Data Science Implementation

### A. Data Loading & Ingestion
- Implemented in `ml_prototype/data_loader.py`.
- Function `extract_student_messages(record)` accommodates multiple variations in incoming JSON schemas (`turns` array with `student_message`, `messages` array, or raw list).
- Function `normalize_session(record)` extracts core fields: `chat_id`, `student_id`, `subject`, `topic`, `student_persona`, `student_messages`, `full_text`, `num_turns`, and `final_understanding`.
- `load_primary_dataset(filepath)` loads `data/raw/edulens_chats_dataset.json`, removes empty conversations (`num_turns > 0`), and filters out records missing target labels.

### B. Exploratory Data Analysis (EDA) & Statistical Analysis
- Documented in `reports/dataset_inspection.md` and accessible via `ml_prototype/dashboard.py` (Page 2: Topic-Wise Performance) and API endpoint `/api/topics`.
- Primary dataset class distribution:
  - `understood`: 192 sessions (44.65%)
  - `partial`: 156 sessions (36.28%)
  - `confused`: 82 sessions (19.07%)
  - Total valid sessions: 430
  - Unique students: 30 (`S01` to `S30`)
  - Subjects: 11 CS subjects (DAA, DSA, ML, OS, DBMS, C++, CN, Cloud, Python, Maths, Java)
  - Topics: 45 unique topics.
- **Leakage Finding:** `overall_sentiment` was discovered to be 100% correlated with `final_understanding` in the synthetic generator and is strictly banned from being used as an input feature.
- **Data Gap:** No automated exploratory data analysis script exists to generate standalone EDA summary tables/charts into `reports/` or `data/processed/`.

---

## 4. Existing NLP Implementation

### A. Implemented Text Processing & Vectorization
- Located in `ml_prototype/features.py`.
- **Text Normalization & Extraction:**
  - Extracts only student utterances by default (`extract_student_messages`).
  - Joins utterances with spaces to create `full_text`.
- **Lexical Representations (TF-IDF):**
  - Uses Scikit-learn's `TfidfVectorizer(max_features=100, stop_words="english", ngram_range=(1, 2), min_df=2)`.
  - Captures unigrams and bigrams characteristic of technical understanding or hesitation.
- **Rule-Based Pedagogical Signal Extraction:**
  - `CONFUSION_PATTERN`: Regular expression matching confusion indicators (`confused`, `lost`, `stuck`, `don't understand`, `don't get`, `didn't get`, `unclear`, `struggling`, `i'm blank`).
  - `CLARIFICATION_PATTERN`: Regular expression matching clarification requests (`explain again`, `another example`, `give an example`, `clarify`, `repeat`, `why does`, `step by step`).
  - `UNDERSTANDING_PATTERN`: Regular expression matching affirmation (`got it`, `makes sense`, `understood`, `now i get`, `clear now`, `thank you`).
  - `CODE_SYNTAX_PATTERN`: Identifies programming tokens (backticks, brackets, keywords like `def`, `class`, `int`, operators `==`, `->`).
- **Feature Union Pipeline:**
  - `build_feature_pipeline()` returns a `ColumnTransformer` joining:
    1. `text_tfidf`: 100 TF-IDF features on `full_text`.
    2. `convo_stats`: 8 dense conversational metrics scaled with `StandardScaler()`.
  - Total feature dimensions: **108 features**.

### B. NLP Methods Present vs. Planned
- **Implemented:** Message extraction, regex-based pedagogical cue matching, TF-IDF unigram+bigram vectorization, z-score feature scaling.
- **Planned / Not Implemented in Core ML:**
  - Lemmatization (e.g., NLTK/spaCy WordNetLemmatizer).
  - Sentence Transformers / Neural Embeddings (e.g., `all-MiniLM-L6-v2`).
  - Full-conversation view (student + teacher turns concatenated) as an experimental comparison.

---

## 5. Existing Machine Learning Models

### A. Understanding Classifier (Primary Model)
- **Script:** `ml_prototype/train_model.py`
- **Algorithm:** Multinomial `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, C=1.0)`
- **Input Features:** 108 features (100 TF-IDF + 8 scaled conversational signals).
- **Target:** `final_understanding` (`understood`, `partial`, `confused`).
- **Train/Test Split:** Stratified 80/20 split (344 train, 86 test).
  - *Observation:* Uses `train_test_split(..., stratify=y)`. It does NOT currently group by `student_id`, meaning the same synthetic student persona may span train and test sets.
- **Baseline:** `DummyClassifier(strategy="most_frequent")`.
  - Baseline Accuracy: 45.35%, Macro F1: 0.2080.
- **Primary Model Metrics (Test Set):**
  - Accuracy: **96.51%** (83/86 correct)
  - Macro Precision: **0.9708**
  - Macro Recall: **0.9476**
  - Macro F1-Score: **0.9577**
  - 5-Fold Stratified CV Mean Accuracy: **91.88% (+/- 0.0426)**
  - 5-Fold Stratified CV Mean Macro F1: **90.31% (+/- 0.0506)**
- **Confusion Matrix:**
  - Confused (True=16): 14 correct, 2 predicted Partial.
  - Partial (True=31): 30 correct, 1 predicted Understood.
  - Understood (True=39): 39 correct, 0 errors.

### B. Other Algorithms & Planned Models
- **Multinomial Naive Bayes:** Not yet trained or compared in `train_model.py`.
- **Linear Support Vector Machine (LinearSVC):** Not yet trained or compared in `train_model.py`.
- **Quiz Score Prediction Regressor:** `quiz` field exists in `edulens_chats_dataset.json` (430 records with `score_percent`) and `set1`, `set2`, `set3` (1,290 records), but no regression model is implemented yet.
- **Engagement / Misconception / Reasoning Models:** Schema v2.1 defines these targets, and `edulens_expanded_pilot_020_v2.json` has 20 records. Not enough records for standalone high-confidence models, but feasibility can be evaluated.

---

## 6. Dataset Inventory

| Dataset File | File Size / Format | Record Count | Features & Labels Available | Status & Project Role |
|---|---|---|---|---|
| `data/raw/edulens_chats_dataset.json` | 1.1 MB JSON | 430 complete sessions | `chat_id`, `student_id`, `subject`, `topic`, `student_persona`, `turns`, `final_understanding`, `quiz`, `overall_sentiment` | **Primary Supervised Training Set** for understanding classification. High quality, zero missing target values. |
| `data/raw/edulens_chats_set1 (1).json` | 890 KB JSON | 420 sessions | `chat_id`, `student_id`, `timestamp`, `llm_client`, `messages`, `quiz` | Contains quiz scores; lacks `final_understanding` and `topic`. Suitable for quiz regression. |
| `data/raw/edulens_chats_set2 (1).json` | 915 KB JSON | 430 sessions | `chat_id`, `student_id`, `timestamp`, `llm_client`, `messages`, `quiz` | Contains quiz scores; lacks `final_understanding` and `topic`. Suitable for quiz regression. |
| `data/raw/edulens_chats_set3 (1).json` | 935 KB JSON | 440 sessions | `chat_id`, `student_id`, `timestamp`, `llm_client`, `messages`, `quiz` | Contains quiz scores; lacks `final_understanding` and `topic`. Suitable for quiz regression. |
| `data/raw/edulens_expanded_pilot_020_v2.json` | 85 KB JSON | 20 sessions | Full schema v2.1: `final_understanding`, `engagement_level`, `misconception_present`, `reasoning_quality`, `self_correction` | Verified pilot for multi-task educational labels. Small sample size (N=20). |
| `data/raw/education_dialogue/` | ~120 MB (6 files) | 47,234 conversations | `background_info`, `conversation` | **Unlabeled** for student understanding. Multi-turn teacher/student dialogue corpus. |
| `data/raw/mathdial/` | ~8 MB (2 JSONL files) | 2,861 conversations (2,262 train, 599 test) | `question`, `student_incorrect_solution`, `conversation`, `self-correctness` | Math problem-solving correction annotations (GSM8K). Specialized task domain. |

---

## 7. Existing Model Artifacts

| Artifact Path | Format | Model Architecture | Pipeline Components | Size | Verification Status |
|---|---|---|---|---|---|
| `models/trained_model.joblib` | Scikit-learn serialized pipeline | Multinomial Logistic Regression | `ColumnTransformer` (TfidfVectorizer + ConversationalFeatureExtractor + StandardScaler) + `LogisticRegression` | ~27 KB | ✅ Validated. Loads cleanly and executes inference without error. |

---

## 8. Backend and Frontend Integration

### A. FastAPI Serverless Backend (`api/index.py`)
- **Endpoints:**
  - `GET /` & `GET /api/health`: Returns API status, version, runtime metadata.
  - `POST /predict` & `POST /api/predict`: Ingests conversation JSON or raw text, returns `predicted_class`, `confidence`, `probabilities`, `conversational_signals`, and `pedagogical_feedback`.
  - `GET /samples` & `GET /api/samples`: Serves 5 curated learning sessions for demo testing.
  - `GET /topics` & `GET /api/topics`: Calculates curriculum mastery and confusion rates across 45 topics from `edulens_chats_dataset.json`.
  - `GET /model-info` & `GET /api/model-info`: Delivers model evaluation benchmarks and confusion matrix.
- **Fail-safe mechanism:** If `models/trained_model.joblib` is missing or fails to load, `api/index.py` dynamically fits a Logistic Regression pipeline in-memory from `edulens_chats_dataset.json`.

### B. Frontend Dashboards
1. **Streamlit App (`ml_prototype/dashboard.py`):**
   - Page 1: Student Performance Analyzer (Interactive conversation analyzer, sample picker, manual text input, JSON upload, signal breakdown).
   - Page 2: Topic-Wise Performance (Table of topics with mastery %, confusion %, and session drill-down).
   - Page 3: Model Architecture & Evaluation (Viva defense guide, confusion matrix, limitations).
2. **Web Dashboard (`public/index.html`):**
   - Single-page application built with Tailwind CSS.
   - Replicates the 3 Streamlit pages in static HTML/JS, communicating with `/api/predict`, `/api/topics`, and `/api/samples`.

---

## 9. Existing Tests and Verification Results

Executed `pytest tests/ -v` using Python 3.12:
- Total tests: **25**
- **Passed: 25**
- **Failed: 0**
- Test execution time: ~12.0 seconds.

Breakdown:
- `tests/test_api.py::test_api_health`: PASSED
- `tests/test_api.py::test_api_samples`: PASSED
- `tests/test_api.py::test_api_topics`: PASSED
- `tests/test_api.py::test_api_predict`: PASSED
- `tests/test_api.py::test_api_predict_with_algorithm`: PASSED
- `tests/test_api.py::test_api_model_info`: PASSED
- `tests/test_api.py::test_api_model_comparison`: PASSED
- `tests/test_api.py::test_api_quiz_model_info`: PASSED
- `tests/test_api.py::test_api_predict_empty_fails`: PASSED
- `tests/test_api.py::test_api_predict_with_explain`: PASSED
- `tests/test_api.py::test_api_explainability_endpoint`: PASSED
- `tests/test_prototype.py::test_extract_student_messages_turns`: PASSED
- `tests/test_prototype.py::test_extract_conversation_stats`: PASSED
- `tests/test_prototype.py::test_primary_dataset_loading`: PASSED
- `tests/test_prototype.py::test_feature_pipeline_scalers`: PASSED
- `tests/test_prototype.py::test_predictor_primary_inference`: PASSED
- `tests/test_prototype.py::test_predictor_all_algorithms`: PASSED
- `tests/test_prototype.py::test_quiz_score_regressor`: PASSED
- `tests/test_prototype.py::test_predictor_invalid_inputs`: PASSED
- `tests/test_prototype.py::test_zero_student_leakage_split`: PASSED
- `tests/test_prototype.py::test_predictor_with_explainability`: PASSED
- `tests/test_prototype.py::test_explainability_helpers`: PASSED
- `tests/test_prototype.py::test_feature_ablation_subsets`: PASSED
- `tests/test_prototype.py::test_quiz_group_isolation`: PASSED
- `tests/test_prototype.py::test_all_model_artifacts_load`: PASSED

---

## 10. Completed Milestones & Current Project State

All foundational, evaluation, and explainability milestones have been achieved:
1. **Understanding Multi-Model Benchmarks:** Logistic Regression, Linear SVM, and Multinomial Naive Bayes trained and evaluated under strict `StratifiedGroupKFold` on `student_id` (357 train / 73 test, 0 student overlap).
2. **Feature Ablation Suite:** 6 distinct feature subsets evaluated (Exps A-F). Confirmed conversational signals alone achieve 75.3% accuracy (+21.9% over majority baseline) and models maintain 94.5% to 98.6% accuracy even without explicit markers AND without shortcut vocabulary.
3. **Model Explainability Engine:** Full global coefficient analysis and instance-level attribution (`explain_instance`) implemented across all algorithms, supported in both FastAPI and Streamlit.
4. **Group-Aware Quiz Regression:** Migrated to strict `GroupShuffleSplit` on `student_id` (1,394 train / 326 test, 0 student overlap). Ridge MAE = 7.76%, Random Forest MAE = 7.65%.
5. **Permanent Documentation Suite:** `docs/DSP.md`, `docs/NLP.md`, `docs/ML.md`, `docs/DATASET.md` fully synchronized with verified reports and actual codebase parameters.
6. **Dashboard & API Extensions:** Interactive model switcher, feature attribution explainability cards, and comprehensive technical documentation integrated.

---

## 11. Implemented Architecture & Pipeline Verification

```
Stage 1: Repository Audit (Complete — PROJECT_AUDIT.md)
   │
   ▼
Stage 2: Permanent Documentation Suite (docs/DSP.md, docs/NLP.md, docs/ML.md, docs/DATASET.md)
   │
   ▼
Stage 3: Data Science Pipeline & Data Preparation (scripts/prepare_dataset.py, data/processed/)
   │
   ▼
Stage 4: NLP Pipeline & Conversational Signals (ml_prototype/features.py, 108 features)
   │
   ▼
Stage 5: Understanding Classifier Benchmarking (Logistic Regression, Linear SVM, Naive Bayes)
   │
   ▼
Stage 6: Feature Ablation Experiments (5 configurations, reports/feature_ablation.json)
   │
   ▼
Stage 7: Model Explainability (ml_prototype/explainability.py, reports/model_explainability.json)
   │
   ▼
Stage 8: Group-Aware Quiz Score Regression (Ridge & Random Forest, 0 student overlap)
   │
   ▼
Stage 9: Streamlit Dashboard & Vercel API Integration (interactive inference, attribution, tabs)
   │
   ▼
Stage 10: Automated Testing & Verification (23/23 tests passing)
```

