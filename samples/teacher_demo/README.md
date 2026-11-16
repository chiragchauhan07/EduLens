# EduLens Teacher Demonstration Dataset

This directory contains eight curated, realistic synthetic student–AI tutor conversations formatted for direct upload and demonstration in **EduLens**.

> [!NOTE]
> **Synthetic Demonstration Data Notice:**  
> All conversations in this collection are synthetically generated educational dialogue samples constructed strictly for technology demonstration, testing, and algorithmic evaluation. They do **not** contain real student records, personal identifiable information (PII), or private educational transcripts.

---

## 1. Overview of Demonstration Samples

EduLens evaluates conversations along two independent dimensions:
1. **Topic Understanding (ML Classifier):** Predicts whether the student's mastery of the topic is `Understood`, `Partially Understood`, or `Confused` using a 108-dimensional feature representation (100 TF-IDF features + 8 conversational metrics) with a trained Multinomial Logistic Regression pipeline.
2. **Expressed Confidence (Linguistic Estimator):** Rule-calibrated heuristic analysis (`High Confidence`, `Moderate Confidence`, `Low Confidence`) derived from explicit conversational markers (assertions, hedges, hesitation tokens, self-corrections, and question ratios).

| File | Topic | Intended Learning Behavior | Predicted Class | Confidence Estimate | Demonstration Highlight |
|---|---|---|---|---|---|
| [`01_strong_understanding.json`](01_strong_understanding.json) | Binary Search | Clear, independent, correct conceptual explanation | **Understood** (99.8%) | **High Confidence** (75.0%) | Shows high understanding paired with high certainty. |
| [`02_partial_understanding.json`](02_partial_understanding.json) | Recursion | Grasps recursive calls but struggles with the base case | **Partially Understood** (53.1%) | **Moderate Confidence** (55.0%) | Illustrates intermediate understanding where the student needs guidance on termination conditions. |
| [`03_conceptual_confusion.json`](03_conceptual_confusion.json) | Pointers in C | Repeated difficulty with memory addresses and dereferencing | **Confused** (43.6%) | **Moderate Confidence** (51.0%) | Shows how persistent struggle and clarification requests shift the model to the confused class. |
| [`04_low_confidence.json`](04_low_confidence.json) | Chemical Bonding | Solid factual knowledge masked by constant hesitation and self-doubt | **Understood** (93.4%) | **Low Confidence** (5.0%) | **Crucial Contrast:** Proves understanding and confidence are decoupled; student knows the facts but lacks self-assurance. |
| [`05_high_confidence.json`](05_high_confidence.json) | Newton's Laws | Assertive, formula-backed explanations with real-world applications | **Understood** (54.1%) | **High Confidence** (95.0%) | Shows strong mathematical and physical reasoning with assertive linguistic phrasing. |
| [`06_learning_improvement.json`](06_learning_improvement.json) | Photosynthesis | Starts with a misconception about $O_2$ release, gets guided, and reaches synthesis | **Understood** (99.0%) | **Moderate Confidence** (68.0%) | Demonstrates dynamic trajectory tracking: initial confusion resolved through pedagogical scaffolding. |
| [`07_misconception.json`](07_misconception.json) | Database Normalization | Student confidently asserts an incorrect concept (storing CSV strings in 1NF) | **Partially Understood** (57.5%) | **High Confidence** (95.0%) | **The "Confidently Wrong" Scenario:** Student expresses high confidence (95%), but model detects partial understanding due to flawed arguments. |
| [`08_mathematical_reasoning.json`](08_mathematical_reasoning.json) | Probability & Combinatorics | Solves dice probability, double-counts outcomes, self-corrects, and validates | **Partially Understood** (51.5%) | **High Confidence** (80.0%) | Demonstrates active problem-solving with visible metacognition and mathematical verification. |

---

## 2. Step-by-Step Teacher Demonstration Sequence

Follow this sequence during a live presentation to showcase the capabilities of EduLens:

### Step 1: Open the Application
- Open the live deployment: [https://edu-lens-b3fq.vercel.app/](https://edu-lens-b3fq.vercel.app/) (or run locally at `http://127.0.0.1:8000`).
- Point out the status badge: **"Learning Analysis • Ready"** with the pulsing green dot, confirming the Scikit-learn inference engine is loaded.

### Step 2: Baseline Demonstration — Strong Understanding
1. In the **Upload Conversation** card, click **Browse JSON file** and select `01_strong_understanding.json`.
2. Click **Analyze Conversation**.
3. **Show the Teacher:**
   - **Primary Classification:** `Understood` (99.8% probability).
   - **Expressed Confidence:** `High Confidence` (75.0%).
   - **Probability Distribution:** Near-zero probability for `Confused`.
   - **Conversational Metrics:** High word count per turn, low hesitation, zero confusion markers.

### Step 3: Pedagogical Contrast — The "Knows but Doubts" Student
1. Upload `04_low_confidence.json` (Topic: Chemical Bonding).
2. Click **Analyze Conversation**.
3. **Explain the Key Pedagogical Finding:**
   - The ML model classifies understanding as **Understood (93.4%)** because the student correctly explains ionic vs. covalent bonds and electronegativity differences.
   - However, the linguistic confidence estimator reports **Low Confidence (5.0%)** because of 11 uncertainty markers ("I think", "maybe", "not sure", "doubt") and 3 confirmation questions ("is that right?").
   - **Takeaway for Teachers:** An automated grading quiz would simply mark the student right, missing that the student suffers from impostor syndrome or low self-efficacy. EduLens flags this nuance immediately.

### Step 4: The "Confidently Wrong" Student
1. Upload `07_misconception.json` (Topic: Database Normalization).
2. Click **Analyze Conversation**.
3. **Explain the Result:**
   - The student speaks with **High Confidence (95.0%)**, asserting that storing comma-separated lists in a single column is "efficient 1NF".
   - The model flags **Partially Understood (57.5%)** with 42.5% Understood, identifying that the student's conceptual grasp of normalization is flawed despite their assertive tone.

### Step 5: Real-Time Confusion and Support Intervention
1. Upload `03_conceptual_confusion.json` (Topic: Pointers in C).
2. Click **Analyze Conversation**.
3. **Show the Teacher:**
   - Classification shifts to **Confused** (43.6% top probability) with elevated clarification requests and zero understanding markers.
   - The **Pedagogical Recommendations** card automatically recommends: *"Provide targeted foundational review on Pointers in C before advancing to complex exercises."*

---

## 3. Methodological Distinction: ML Understanding vs. Expressed Confidence

When demonstrating to faculty or technical reviewers, emphasize the distinct architectural roles of the two systems:

```
┌──────────────────────────────────────────────────────────────────┐
│                         EduLens Analysis                         │
└────────────────┬────────────────────────────────┬────────────────┘
                 │                                │
                 ▼                                ▼
┌─────────────────────────────────┐  ┌───────────────────────────────┐
│  ML Understanding Classifier    │  │ Expressed Confidence Estimator│
├─────────────────────────────────┤  ├───────────────────────────────┤
│ • Model: Multinomial Logistic   │  │ • Approach: Rule-Calibrated   │
│   Regression Pipeline           │    Linguistic Heuristic          │
│ • Features: 108 dense inputs    │  │ • Input: Explicit dialogue    │
│   (100 TF-IDF + 8 dense stats)  │    markers (hedges, assertions)  │
│ • Target: Cognitive Mastery     │  │ • Target: Linguistic Tone     │
│   (Understood / Partial /       │    (High / Moderate / Low)       │
│    Confused)                    │  │ • Output: Transparent 0-100%  │
│ • Training: Supervised college  │    score with signal audit trail │
│   chat corpus (430 sessions)    │                                  │
└─────────────────────────────────┘  └───────────────────────────────┘
```

### Important Ethical & Technical Disclaimer:
- **Expressed Confidence $\neq$ Internal Psychometric State:**  
  EduLens measures *expressed linguistic certainty* in the text (assertions versus hedges and hesitations). It does **not** claim to read the student's mind or perform clinical psychometric diagnosis. A student might be culturally modest and use hedges while having high internal confidence, or conversely display bluster while hiding internal confusion.
- **Explainability:**  
  Every confidence score provides an expandable breakdown of raw counts (uncertainty count, certainty count, hesitation markers, self-correction frequency, and confirmation requests).

---

## 4. JSON Format Specification

Every demonstration file strictly adheres to the schema:

```json
{
  "topic": "String representing the subject/concept",
  "messages": [
    {
      "role": "assistant",
      "content": "Tutor utterance..."
    },
    {
      "role": "student",
      "content": "Learner utterance..."
    }
  ]
}
```

- Roles must alternate between `"student"` and `"assistant"`.
- Text is stored under the `"content"` key.
- Payloads are well below the system's 5 MB limit (each demo file is ~2–4 KB).
