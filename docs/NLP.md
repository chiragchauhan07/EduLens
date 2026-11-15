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

## 5. Input and Output Example

### Input (Raw Student Utterances):
```json
[
  "can we go through binary search algorithm in DAA ?",
  "what is the recurrence relation and best / worst case time complexity ?",
  "so T(n) = T(n/2) + O(1) gives O(log n) using Master Theorem Case 2, right?",
  "got it! When array is already sorted, mid = low + (high - low) / 2 prevents integer overflow.",
  "understood completely, that makes sense now! thank you!"
]
```

### Extracted Signals:
- `num_turns`: 5.0
- `total_words`: 64.0
- `avg_turn_length`: 12.8
- `question_count`: 3.0
- `confusion_markers`: 0.0
- `clarification_markers`: 0.0
- `understanding_markers`: 4.0 (`got it`, `understood`, `makes sense`, `thank you`)
- `code_indicators`: 2.0

### Vector Output:
Dense vector of dimension 108 (100 TF-IDF weights + 8 scaled metrics).
Ready for ingestion by Logistic Regression, Naive Bayes, or Linear SVM.

