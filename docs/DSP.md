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

*Scaling:* The 8 dense conversational features are scaled using `StandardScaler()` fitted strictly on the training partition.

### B. Lexical TF-IDF Vocabulary Features (100 Features)
- Vectorizer: `TfidfVectorizer(max_features=100, stop_words="english", ngram_range=(1, 2), min_df=2)`
- Evaluated on student-only concatenated dialogue (`full_text`).

---

## 6. Data Science Libraries & Dependencies

- **Pandas (>=2.0.0):** DataFrame normalization, aggregation, topic summary computation.
- **NumPy (>=1.24.0):** Numerical matrix operations, feature concatenation.
- **Scikit-learn (1.5.2):** `ColumnTransformer`, `StandardScaler`, `TfidfVectorizer`, `Pipeline`.
- **Joblib (>=1.3.0):** High-efficiency pipeline serialization.

