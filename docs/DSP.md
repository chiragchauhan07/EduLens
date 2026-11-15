# Data Science Pipeline (DSP) — EduLens

> **Status:** Active Reference & Permanent Technical Record  
> **Module Responsibility:** Data Collection, Ingestion, Normalization, EDA, and Feature Transformation  
> **Last Synchronized:** November 2026  

---

## 1. Data Science Architecture

EduLens processes unstructured educational dialogues into structured numerical matrices for machine learning through a deterministic, reproducible Data Science pipeline.

```text
Raw Conversational JSON
          │
          ▼
   [Data Ingestion] ──── data_loader.py / load_primary_dataset()
          │
          ▼
 [Boundary Normalization] ── normalize_session() / extract_student_messages()
          │
          ▼
[Exploratory Data Analysis] ── statistical distributions, class balances, length metrics
          │
          ▼
 [Data Cleaning & Filtering] ── drop zero-turn dialogues, validate target labels
          │
          ▼
[Feature Extraction & Scaling] ── features.py / build_feature_pipeline()
          │
          ▼
Prepared Feature Matrix (X, y) ── ready for Scikit-learn estimators
```

---

## 2. Data Ingestion & JSON Loading

### Ingestion Source Files
The primary supervised data is stored in `data/raw/edulens_chats_dataset.json`.  
Supplementary multi-turn session files include:
- `data/raw/edulens_chats_set1 (1).json` (420 sessions)
- `data/raw/edulens_chats_set2 (1).json` (430 sessions)
- `data/raw/edulens_chats_set3 (1).json` (440 sessions)

### Ingestion Functions
- `ml_prototype.data_loader.extract_student_messages(record)`:
  Handles structural variations across data generation phases:
  - Turn objects containing `student_message` key (`record["turns"]`).
  - Message objects containing `student_message` key (`record["messages"]`).
  - Flat lists of student utterance strings (`record["student_messages"]`).
- `ml_prototype.data_loader.normalize_session(record)`:
  Maps raw dictionaries into standardized internal dictionaries:
  - `chat_id`: Unique session identifier (e.g., `CHAT_0001`).
  - `student_id`: Student cluster identifier (e.g., `S01` to `S30`).
  - `subject`: Academic subject name (e.g., `DSA`, `Python`, `ML`).
  - `topic`: Granular topic name (e.g., `Binary Search`, `Pointers`).
  - `student_persona`: Simulated learner profile (e.g., `beginner`, `skeptical`).
  - `student_messages`: Cleaned list of student utterance strings.
  - `full_text`: Space-concatenated single string of all student utterances.
  - `num_turns`: Total count of student dialogue turns.
  - `final_understanding`: Supervised label (`understood`, `partial`, `confused`).

---

## 3. Data Cleaning & Validation

1. **Filtering Empty Sessions:** Sessions with `num_turns == 0` or missing conversational text are excluded.
2. **Missing Value Handling:** Records where `final_understanding` is null or undefined are excluded from supervised classification.
3. **Leakage Elimination:**
   - In synthetic data generation, `overall_sentiment` was found to be tightly correlated with understanding outcome. It is strictly excluded from feature extraction.
   - Post-session quiz score fields (`quiz.score_percent`) are reserved for downstream regression tasks and never passed into the understanding classifier.
   - Session identifiers (`chat_id`, `student_id`) and timestamps are purged from predictive feature vectors.

---

## 4. Exploratory Data Analysis & Statistical Distributions

### Class Distribution (Primary Dataset, N = 430)
- **`understood`**: 192 sessions (44.65%)
- **`partial`**: 156 sessions (36.28%)
- **`confused`**: 82 sessions (19.07%)
- **Class Imbalance Handling:** Addressed during model training using inverse class weighting (`class_weight="balanced"`) and macro-averaged evaluation metrics.

### Academic Subject Coverage (11 Subjects)
- DAA: 80 sessions
- DSA: 80 sessions
- Machine Learning (ML): 63 sessions
- Operating Systems (OS): 50 sessions
- DBMS: 40 sessions
- C++: 27 sessions
- Computer Networks (CN): 27 sessions
- Cloud Computing: 18 sessions
- Python: 18 sessions
- Mathematics: 18 sessions
- Java: 9 sessions

### Conversational Length Metrics
- **Mean Student Turns per Session:** 4.8 turns (std: 1.1 turns; range: 3 to 7 turns).
- **Mean Words per Student Turn:** 14.2 words.
- **Mean Total Words per Session:** 68.4 words.

---

## 5. Feature Engineering

The feature extraction architecture in `ml_prototype/features.py` produces 108 total features using a Scikit-learn `ColumnTransformer`:

### A. Dense Conversational Signals (8 Features)
Extracted by `ConversationalFeatureExtractor`:
1. `num_turns`: Count of student turns.
2. `total_words`: Total word count across all student utterances.
3. `avg_turn_length`: Mean words per turn.
4. `question_count`: Count of question marks (`?`) in student text.
5. `confusion_markers`: Frequency of confusion regex matches (`confused`, `lost`, `stuck`, `don't understand`, `unclear`, `didn't get`).
6. `clarification_markers`: Frequency of clarification request regex matches (`explain again`, `another example`, `clarify`, `repeat`, `why does`).
7. `understanding_markers`: Frequency of comprehension confirmation regex matches (`got it`, `makes sense`, `understood`, `now i get`, `clear now`).
8. `code_indicators`: Frequency of code/programming syntax indicators (backticks, braces, keywords like `def`, `class`, `int`, operators).

*Scaling & Non-Negativity:*
- **Standard Scaling:** For Logistic Regression and Linear SVM, the 8 dense conversational features are centered and normalized using `StandardScaler()` fitted strictly on the training partition.
- **MinMax Scaling:** For Multinomial Naive Bayes, all features are normalized via `MinMaxScaler(feature_range=(0, 1))` to guarantee non-negative inputs ($x_i \ge 0$), as required by the multinomial distribution.

### B. Lexical TF-IDF Vocabulary Features (100 Features)
- Vectorizer: `TfidfVectorizer(max_features=100, stop_words="english", ngram_range=(1, 2), min_df=2)`
- Evaluated on student-only concatenated dialogue (`full_text`).

### C. Pipeline Flexibility & Ablation Support
The factory function `build_feature_pipeline(scaler, max_tfidf_features, include_tfidf, include_convo, convo_features, custom_stop_words)` in `ml_prototype/features.py` enables:
- **Exp A:** Lexical TF-IDF only (`include_convo=False`, 100 features).
- **Exp B:** Dense conversational signals only (`include_tfidf=False`, 8 features).
- **Exp C:** Full combined feature space (108 features).
- **Exp D:** Ablated signals without explicit markers (`convo_features=[...]`, 106 features).
- **Exp E:** Without dense conversational signals (100 features).
- **Exp F:** Ablated markers and shortcut vocabulary filtering (`custom_stop_words=[...]`, 106 features).

---

## 6. Group-Aware Splitting Methodology

To eliminate synthetic student persona leakage, all cross-validation and benchmark splits enforce strict student grouping:
1. **Understanding Classification (430 sessions, 30 students):**
   - Partitioned via `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on `student_id`.
   - Training partition: 357 sessions across 24 students.
   - Held-out test partition: 73 sessions across 6 students.
   - **Student overlap: 0 students.**
2. **Quiz Score Regression (1,720 sessions, 40 students):**
   - Partitioned via `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` on `student_id`.
   - Training partition: 1,394 sessions across 32 students.
   - Held-out test partition: 326 sessions across 8 students.
   - **Student overlap: 0 students.**

---

## 7. Modular DSP Pipeline Implementation (`ml_prototype/dsp.py`)

To support clean production integration with external MCP connectors and UI frontends, the DSP logic is consolidated into a dedicated, modular pipeline component in `ml_prototype/dsp.py`:

### Key Capabilities:
1. **Schema Validation (`validate_session_json`):**
   - Verifies presence of essential conversational dialogue keys (`turns`, `messages`, `student_messages`, or multi-topic payloads).
   - Validates turn structures, detects malformed JSON payloads, and returns descriptive error messages with missing key diagnostics.
2. **Structural Normalization (`normalize_session_full`):**
   - Ingests raw heterogeneous payloads (Schema v1, v2, v2.1, flat utterance arrays, or raw text blocks).
   - Extracts separated student and teacher dialogue turns.
   - Cleans whitespace, handles Unicode artifacts, and removes duplicate adjacent turns.
3. **Conversational Dynamics Analysis (`extract_dsp_features`):**
   - `student_turn_count` & `teacher_turn_count`
   - `student_word_count` & `teacher_word_count`
   - `student_to_teacher_word_ratio` & `turn_ratio`
   - `turn_alternations`: Number of active speaker switches between student and instructor.
   - `student_questions` & `teacher_questions`
4. **Data Integrity & Group Isolation (`verify_group_isolation`):**
   - Deterministically verifies that no `student_id` exists in both training and test partitions.

---

## 8. Data Science Libraries & Dependencies

- **Pandas (>=2.0.0):** DataFrame normalization, aggregation, topic summary computation.
- **NumPy (>=1.24.0):** Numerical matrix operations, feature concatenation.
- **Scikit-learn (1.5.2):** `ColumnTransformer`, `StandardScaler`, `MinMaxScaler`, `TfidfVectorizer`, `Pipeline`.
- **Joblib (>=1.3.0):** High-efficiency pipeline serialization.

