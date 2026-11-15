# Dataset Documentation (DATASET) — EduLens

> **Status:** Active Reference & Permanent Technical Record  
> **Module Responsibility:** Data Provenance, Schemas, Annotation Protocols, Dataset-to-Model Mapping  
> **Last Synchronized:** November 2026  

---

## 1. Dataset Inventory & Summary

EduLens maintains an explicit separation between primary supervised training datasets, supplementary corpora, and external research benchmarks:

| Dataset Name | File Path | Record Count | Supervised Target(s) | Role in EduLens |
|---|---|---|---|---|
| **EduLens Primary Chats** | `data/raw/edulens_chats_dataset.json` | 430 complete sessions | `final_understanding`, `quiz.score_percent` | **Primary Supervised Training & Test Set** for understanding classification. |
| **EduLens Set 1** | `data/raw/edulens_chats_set1 (1).json` | 420 sessions | `quiz.score_percent` | Secondary corpus for quiz score regression experiments. |
| **EduLens Set 2** | `data/raw/edulens_chats_set2 (1).json` | 430 sessions | `quiz.score_percent` | Secondary corpus for quiz score regression experiments. |
| **EduLens Set 3** | `data/raw/edulens_chats_set3 (1).json` | 440 sessions | `quiz.score_percent` | Secondary corpus for quiz score regression experiments. |
| **EduLens Pilot v2.1** | `data/raw/edulens_expanded_pilot_020_v2.json` | 20 sessions | `final_understanding`, `engagement_level`, `misconception_present`, `reasoning_quality` | Schema validation and exploratory multi-task modeling. |
| **Education Dialogue** | `data/raw/education_dialogue/*.json` | 47,234 conversations | *None (Unlabeled)* | Unsupervised dialogue flow / linguistic exploration only. |
| **MathDial** | `data/raw/mathdial/*.jsonl` | 2,861 conversations | `self-correctness` | Domain benchmark on human math problem-solving dialogues (GSM8K). Kept separate to prevent label pollution. |

---

## 2. Primary Dataset Specification (`edulens_chats_dataset.json`)

- **Format:** JSON array of 430 session dictionaries.
- **Data Provenance:** Synthesized with Claude using detailed pedagogical system prompts simulating distinct student personas and learning struggles.
- **Unique Students:** 30 simulated students (`S01` to `S30`), with 10 to 18 sessions per student.
- **Top-Level Record Schema:**

| Field | Type | Description | Permitted Values / Range |
|---|---|---|---|
| `chat_id` | String | Unique session identifier | `CHAT_0001` to `CHAT_0430` |
| `student_id` | String | Simulated student identifier | `S01` to `S30` (used for group-aware splitting) |
| `subject` | String | Academic subject | 11 Computer Science subjects (DAA, DSA, ML, OS, etc.) |
| `topic` | String | Granular concept topic | 45 topics (e.g., "Pointers", "Virtual Memory") |
| `student_persona` | String | Learning style simulation | `beginner`, `skeptical`, `exam`, `casual`, `curious`, `debug` |
| `turns` | Array | Sequence of dialogue turns | 3 to 7 turns per session |
| `final_understanding` | String | **Primary Supervised Label** | `understood`, `partial`, `confused` |
| `quiz` | Object | Post-session assessment data | `{"num_questions": int, "correct": int, "score_percent": int}` |
| `overall_sentiment` | String | Simulated overall mood | `positive`, `mixed`, `negative` (**Leakage Warning: Excluded from ML**) |

- **Turn Object Schema (`turns[i]`):**
  - `turn`: Sequential integer (1-indexed).
  - `student_message`: Student dialogue utterance text.
  - `question_type`: Pedagogical question category (e.g., `clarification`, `why`, `definition`).
  - `cognitive_level`: Bloom's taxonomy level (`Remember`, `Understand`, `Apply`, `Analyze`).
  - `sentiment`: Turn-level sentiment.
  - `confusion_flag`: Boolean indicating whether struggle was flagged on this turn.

---

## 3. Dataset-to-Model Mapping Table

| Model Task | Compatible Datasets | Target Variable | Excluded Fields (Leakage Prevention) |
|---|---|---|---|
| **Student Understanding Classification** | `edulens_chats_dataset.json` (430 sessions) | `final_understanding` | `overall_sentiment`, `quiz.*`, `turns[].sentiment`, `turns[].confusion_flag`, metadata IDs |
| **Quiz Score Prediction (Regression)** | `edulens_chats_dataset.json` + `set1` + `set2` + `set3` (1,720 sessions) | `quiz.score_percent` | `final_understanding`, metadata IDs, timestamps |
| **Multi-Task Pilot Feasibility** | `edulens_expanded_pilot_020_v2.json` (20 sessions) | `engagement_level`, `misconception_present`, `reasoning_quality` | Metadata IDs, teacher rationale |

---

## 4. Dataset Rules & Integrity Guidelines

1. **No Incompatible Label Mixing:** MathDial labels (`self-correctness`) represent whether an elementary arithmetic riddle was resolved. They must never be merged into EduLens CS conceptual understanding labels.
2. **Never Supervise with Unlabeled Corpora:** The 47,234 Education Dialogue conversations contain no understanding ground truth and must not be used for supervised training.
3. **Immutability of Raw Data:** `data/raw/` is read-only. Cleaned, aggregated, or transformed files are saved strictly to `data/processed/`.
4. **Group-Aware Splitting Methodology:**
   To eliminate synthetic student persona leakage, all evaluations enforce strict group isolation by `student_id`:
   - **Classification Partitioning (`StratifiedGroupKFold`, 5 splits):**
     * Total Sessions: 430 across 30 students (`S01` to `S30`).
     * Train Partition: 357 sessions across 24 students (83.02%).
     * Test Partition: 73 sessions across 6 students (16.98%).
     * Student Overlap: **0 students** (100% strict group isolation).
   - **Quiz Regression Partitioning (`GroupShuffleSplit`, 80/20):**
     * Total Sessions: 1,720 across 40 students (`S01` to `S40`).
     * Train Partition: 1,394 sessions across 32 students (81.05%).
     * Test Partition: 326 sessions across 8 students (18.95%).
     * Student Overlap: **0 students** (100% strict group isolation).

