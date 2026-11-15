# Natural Language Processing (NLP) — EduLens

> **Status:** Active Reference & Permanent Technical Record  
> **Module Responsibility:** Conversational Text Extraction, Lexical Transformation, Signal Parsing, Vectorization  
> **Last Synchronized:** November 2026  

---

## 1. NLP Pipeline Overview

The EduLens NLP pipeline converts unstructured student conversational dialogue into fixed-dimensional numerical feature vectors suitable for statistical machine learning estimators.

```text
Raw Conversation JSON
          │
          ▼
 [Utterance Extraction] ── extract_student_messages()
          │
          ▼
  [Text Normalization] ── lowercasing, whitespace stripping, turn joining
          │
          ├──► [Pedagogical Signal Extraction] ── Regular Expressions (8 dense metrics)
          │
          └──► [Lexical Vectorization] ── TF-IDF N-grams (100 lexical features)
          │
          ▼
   [Feature Union] ── ColumnTransformer combining Scaled Signals + TF-IDF
          │
          ▼
Feature Matrix: Shape (N, 108)
```

---

## 2. Conversation Extraction & Normalization

### Student Message Isolation
By default, EduLens operates strictly on **student-generated utterances**. This ensures that the model learns student understanding from the student's own words rather than echoing the teacher/tutor's linguistic sophistication.

```python
# Extraction logic in ml_prototype/data_loader.py
def extract_student_messages(record: Dict[str, Any]) -> List[str]:
    # Extracts student messages across schema variants ('turns', 'messages', or flat list)
    ...
```

### Text Normalization & Assembly
1. Strips leading and trailing whitespace from each turn.
2. Joins all student turns in sequential order with a single space to construct `full_text`.
3. Standardizes apostrophes and punctuation for regex matching.

---

## 3. Pedagogical & Conversational Signal Extraction

Unlike standard sentiment analyzers, EduLens uses targeted pedagogical regular expressions to capture learning behaviors:

| Signal Name | Pattern Logic | Educational Motivation |
|---|---|---|
| `confusion_markers` | `\b(confus\w*\|lost\|stuck\|don'?t\s+understand\|don'?t\s+get\|not\s+getting\|didn'?t\s+get\|doubt\|hard\|unclear\|struggl\w*\|what\s+does\s+that\s+mean\|huh\|i'?m\s+blank\|wait\s+what)\b` | Identifies active cognitive disequilibrium and struggle. |
| `clarification_markers` | `\b(explain\s+again\|another\s+example\|give\s+an\s+example\|can\s+you\s+show\|clarif\w*\|repeat\|difference\s+between\|why\s+does\|how\s+come\|step\s+by\s+step)\b` | Detects requests for scaffolded instruction or elaboration. |
| `understanding_markers` | `\b(got\s+it\|makes?\s+sense\|understood\|now\s+i\s+get\|i\s+see\|clear\s+now\|so\s+basically\|that\s+means\|ah\s+okay\|oh\s+okay\|thank\s+you\|thanks\|perfect)\b` | Measures explicit student self-reported comprehension and resolution. |
| `code_indicators` | `(`{1,3}\|[{};]\|\b(def\|class\|return\|int\|float\|void\|print\|import\|for\|while)\b\|==\|!=\|->)` | Quantifies technical engagement and code articulation in CS topics. |
| `question_count` | Count of `?` characters | Approximates student inquiry frequency and active curiosity. |
| `avg_turn_length` | Total words divided by number of turns | Differentiates terse, low-effort replies from detailed explanations. |

---

## 4. Lexical Vectorization (TF-IDF)

### Configuration
- **Estimator:** Scikit-learn `TfidfVectorizer`
- **Max Features:** 100
- **N-gram Range:** `(1, 2)` (Unigrams and Bigrams)
- **Stop Words:** Built-in English stop words
- **Min Document Frequency (`min_df`):** 2 documents
- **Sublinear TF Scaling:** False (standard raw term frequency weighted by IDF)

### Why 100 Features?
In educational conversations spanning ~430 records, a compact vocabulary prevents the curse of dimensionality and ensures that coefficients remain interpretable without severe over-parameterization.

---

## 5. Detailed Prediction Walkthrough Trace (7 Stages)

The following end-to-end trace illustrates how a real learning session (`CHAT_0001`) from `data/raw/edulens_chats_dataset.json` travels through the 7 stages of the EduLens NLP and ML pipeline:

### Stage 1: Raw JSON Ingestion
```json
{
  "chat_id": "CHAT_0001",
  "subject": "C++",
  "topic": "Pointers and Memory Management",
  "student_persona": "confused",
  "turns": [
    {
      "turn": 1,
      "student_message": "i don't understand pointers in C++ at all, it's so confusing",
      "teacher_message": "Pointers can be tricky! Think of a pointer as a variable that holds a memory address..."
    },
    {
      "turn": 2,
      "student_message": "what is the difference between * and &? I'm completely lost",
      "teacher_message": "Good question. & is the 'address-of' operator, while * is the dereference operator..."
    },
    {
      "turn": 3,
      "student_message": "why does int* ptr = &x give a segmentation fault when I dereference it later?",
      "teacher_message": "A segfault usually happens when you dereference an uninitialized or invalid pointer..."
    }
  ],
  "final_understanding": "confused"
}
```

### Stage 2: Student Message Extraction (`extract_student_messages`)
Teacher messages are filtered out; only student utterances are retained:
1. `Turn 1:` *"i don't understand pointers in C++ at all, it's so confusing"*
2. `Turn 2:` *"what is the difference between * and &? I'm completely lost"*
3. `Turn 3:` *"why does int* ptr = &x give a segmentation fault when I dereference it later?"*

### Stage 3: Text Normalization
- **Full Text String (41 words):**
  > `"i don't understand pointers in C++ at all, it's so confusing what is the difference between * and &? I'm completely lost why does int* ptr = &x give a segmentation fault when I dereference it later?"`

### Stage 4: Pedagogical Conversational Feature Extraction (8 Dense Metrics)
- `num_turns`: 3.0
- `total_words`: 41.0
- `avg_turn_length`: 13.67 words
- `question_count`: 3.0 (questions in each turn)
- `confusion_markers`: 2.0 (matches: `"don't understand"`, `"confusing"`, `"lost"`)
- `clarification_markers`: 0.0
- `understanding_markers`: 0.0
- `code_indicators`: 1.0 (matches: `"int*"` operator syntax)

### Stage 5: TF-IDF Transformation (100 Vocabulary Features)
Active non-zero TF-IDF terms in the normalized vocabulary:
- `confused`: 0.521
- `difference`: 0.487
- `lost`: 0.514
- `pointers`: 0.476

### Stage 6: 108-Dimensional Feature Vector Assembly
- **Dimensions [0 to 7] (Scaled Conversational Metrics):** Centered and standardized using `StandardScaler` trained on 357 training sessions:
  `[ -1.63, -0.98, -0.15, 0.52, 2.84, -0.81, -1.30, -0.96 ]`
- **Dimensions [8 to 107] (Lexical TF-IDF Vector):** 100-dimensional sparse-to-dense float array.
- **Combined Shape:** $(1, 108)$.

### Stage 7: ML Inference & Explainability Output
- **Model Ingested:** Multinomial Logistic Regression (`models/understanding/logistic_regression/model.joblib`)
- **Predicted Class:** `confused` (Ground Truth: `confused` — ✅ Correct Match)
- **Class Probabilities:**
  - `confused`: **71.93%** (Confidence)
  - `partial`: **27.84%**
  - `understood`: **0.23%**
- **Predicted Quiz Score:** **33.64%** (Reflects struggling performance)
- **Top Local Evidence Drivers:**
  - `+3.251` from absence of `understanding_markers` (pushing away from `understood`)
  - `+1.880` from low total word volume
  - `+0.451` from lexical token `don`
  - `-1.135` from high `confusion_markers` (pushing strongly towards `confused`)
- **Pedagogical Feedback:** *"The student expressed significant doubt, repeated questions, or struggled to apply the core concepts. Recommend 1-on-1 human tutor review or foundational review."*

