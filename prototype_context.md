# EduLens Mini — ML Prototype Context & Practical Showcase Guide

> **Document Status:** Verified and synchronized with EduLens Mini Practical Implementation  
> **Target Audience:** Practical Examiners, Evaluators, Engineering Faculty, Student Presenters  
> **Version:** EduLens Mini Final Practical Showcase  

---

## 1. What EduLens Mini Is

**EduLens Mini** is a specialized, lightweight student learning intelligence application for educational dialogue analysis.

- **EduLens Mini is NOT an AI tutor.** It does not teach students or answer tutoring questions.
- Students learn using external AI assistants (ChatGPT, Claude, Socratic tutors).
- Students submit their chat logs to EduLens Mini, which extracts quantifiable, interpretable conversational evidence to evaluate demonstrated understanding, detect learning confusion, and estimate expressed learning confidence.

---

## 2. Practical Showcase Objectives Delivered

1. **Robust Multi-Schema Ingestion**: Ingests educational dialogues across varying schemas (messages list, turns array, separate student/tutor blocks).
2. **Leakage-Free Feature Engineering (108 Features)**:
   - 100 Lexical TF-IDF n-grams (1, 2) capturing domain vocabulary and causal reasoning markers.
   - 8 Conversational & Pedagogical Signals capturing dialogue structure, question frequency, confusion markers, affirmations, and code indicators.
3. **Primary Interpretable ML Model**:
   - Multinomial Logistic Regression (`class_weight='balanced'`, `max_iter=1000`, `C=1.0`).
   - Achieves **95.89% Test Accuracy** and **0.9399 Macro F1-score** on 5-fold Stratified Group K-Fold split.
4. **Transparent Expressed Confidence Estimation**:
   - Rule-calibrated discourse analysis based on uncertainty hedges, assertions, and self-repair cues.
   - Explicitly distinguished from ML probabilities and internal psychological conviction.
5. **Fast, Lightweight Inference**:
   - Single-file and multipart JSON uploads processed in under 50 milliseconds.
   - Zero external LLM API keys or cloud dependencies needed during inference.
6. **Modern Apple/Google-Inspired Web Dashboard**:
   - Designed with Google Stitch MCP.
   - Clean drag-and-drop JSON upload, real-time results visualization, probability ring, class distribution breakdown, and transparent methodology modal.
7. **Automated Verification**:
   - 57 passing automated unit and integration tests (`pytest tests/ -v`).

---

## 3. Dataset Audit & Provenance

The repository contains three dataset sources in `data/raw/`:

1. **Claude-Generated EduLens Dataset** (`data/raw/edulens_chats_dataset.json`):
   - **Primary Supervised Dataset** (430 sessions, 11 Computer Science subjects, 30 distinct student personas).
   - Annotated with ground-truth `final_understanding` classes: `understood` (230), `partial` (116), `confused` (84).
   - Grouped by `student_id` during cross-validation to guarantee zero persona leakage.
2. **Education Dialogue Dataset** (`data/raw/education_dialogue/`):
   - 47,234 conversations (40,000 train, 7,234 eval) from Gemini Ultra RLHF paper. Preserved for research.
3. **MathDial Dataset** (`data/raw/mathdial/`):
   - 2,861 math tutoring dialogues on GSM8K word problems. Preserved for reference.
4. **Pilot v2.1 Dataset** (`data/raw/edulens_expanded_pilot_020_v2.json`):
   - 20 pilot sessions with preliminary confidence annotations (10 annotated, 10 unknown).
   - Demonstrated that large-scale human confidence labels do not yet exist, justifying the rule-calibrated linguistic indicator approach.

---

## 4. NLP Preprocessing & Feature Extraction

### The 108-Dimensional Feature Vector:
```text
Conversation Messages -> Extract Student Messages -> Concatenate -> Full Text
                                                                       │
           ┌───────────────────────────────────────────────────────────┴────────────────────────┐
           ▼                                                                                    ▼
[TF-IDF Sub-Pipeline]                                                              [Conversational Sub-Pipeline]
TfidfVectorizer(max_features=100, ngram_range=(1,2), stop_words='english')        ConversationalFeatureExtractor() -> StandardScaler()
  • 100 lexical n-grams                                                               • num_turns (Student message count)
  • Domain vocabulary                                                                 • total_words (Student word count)
  • Causal markers ("because", "therefore")                                           • avg_turn_length (Words per turn)
                                                                                      • question_count (Inquiry frequency)
                                                                                      • confusion_markers (doubt, lost, stuck)
                                                                                      • clarification_markers (explain again)
                                                                                      • understanding_markers (got it, makes sense)
                                                                                      • code_indicators (syntax, def, return)
           │                                                                                    │
           └────────────────────────────────────────┬───────────────────────────────────────────┘
                                                    ▼
                                     ColumnTransformer Feature Union
                                         (108 Numerical Features)
                                                    │
                                                    ▼
                                     Logistic Regression Classifier
```

---

## 5. Expressed Confidence Estimation Methodology

### Distinction Between Model Probability & Expressed Confidence:
- **Model Probability** ($P(\hat{y} \mid X)$): The classifier's posterior statistical certainty that the dialogue belongs to the predicted understanding class.
- **Expressed Confidence**: A linguistic measure of how assertively or tentatively the student phrased their thoughts in the conversation.

### Formula:
$$\text{Score} = 55.0 + 9.0 \times \text{Certainty} + 7.0 \times \text{Explanations} + 5.0 \times \text{Self-Corrections} - 8.0 \times \text{Uncertainty} - 5.0 \times \text{Hesitation} - 4.0 \times \text{Confirmations} - 9.0 \times \text{Hint-Dependence}$$

Clamped to $[5.0, 95.0]$:
- **High Confidence**: Score $> 68.0\%$
- **Moderate Confidence**: $40.0\% \le \text{Score} \le 68.0\%$
- **Low Confidence**: Score $< 40.0\%$

---

## 6. Viva Voce / Examiner Defense Q&A

### Q1: Why use Logistic Regression instead of a Deep Learning model or LLM API?
> **Answer:** In educational analytics, explainability, predictability, and local execution speed are paramount. Logistic Regression with balanced weighting provides calibrated posterior probabilities, runs in $< 50$ ms without GPU hardware, achieves $95.89\%$ test accuracy, and exposes exact feature weights for inspection. It does not suffer from hallucinations or unpredictable API rate limits.

### Q2: How did you prevent data leakage during model training?
> **Answer:** We used `StratifiedGroupKFold(n_splits=5)` grouped strictly by `student_id`. This guarantees that utterances from the same student persona never appear in both the training and test splits simultaneously, ensuring the model evaluates conceptual comprehension rather than memorizing individual writing styles.

### Q3: Why is confidence estimation not presented as an ML model?
> **Answer:** We conducted an audit of available educational datasets. The primary 430-session dataset contains zero validated student confidence labels, and the pilot dataset contains only 10 usable labeled sessions. Training a classifier on 10 samples would be statistically unscientific and prone to overfitting. We chose scientific integrity: implemented a transparent, rule-calibrated linguistic marker score and labeled it as a **linguistic confidence estimate**.

### Q4: How does the application handle invalid or malicious inputs?
> **Answer:** The `conversation_parser` validates JSON syntax, caps upload file size at 5MB, verifies the presence of student utterances, and rejects teacher-only or empty dialogues with descriptive 400 Bad Request error messages.
