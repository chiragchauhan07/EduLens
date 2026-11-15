# Machine Learning (ML) — EduLens

> **Status:** Active Reference & Permanent Technical Record  
> **Module Responsibility:** Model Architecture, Training, Group-Aware Cross-Validation, Feature Ablation, Explainability, Evaluation Benchmarks  
> **Last Synchronized:** November 2026  

---

## 1. Problem Formulation

EduLens formulates student learning evaluation as a **supervised multi-class classification** task:

$$\hat{y} = f(\mathbf{x}), \quad \hat{y} \in \{\text{understood}, \text{partial}, \text{confused}\}$$

Where $\mathbf{x} \in \mathbb{R}^{108}$ is the feature vector extracted strictly from student conversational text and structural dialogue patterns.

### Target Classes & Pedagogical Meaning

| Label | Pedagogical Definition | Recommended Action |
|---|---|---|
| **`understood`** | Student demonstrates solid conceptual comprehension, articulates ideas accurately, confirms resolution with evidence. | Advance to higher-level or synthesis problems. |
| **`partial`** | Student grasps basic premise but exhibits lingering ambiguity, hesitates, or requests repeated scaffolding. | Targeted practice and concrete edge-case exercises. |
| **`confused`** | Student expresses active disequilibrium, asks circular questions, or fails to apply fundamental concepts. | Recommend human instructor review or foundational tutoring. |

---

## 2. Algorithms Benchmarked

### Model 1: Multinomial Logistic Regression (Primary Baseline)
- **Mathematical Principle:** Models log-odds of each class as a linear combination of features using the softmax link function:
  $$P(Y = c \mid \mathbf{x}) = \frac{e^{\mathbf{w}_c^T \mathbf{x} + b_c}}{\sum_{k} e^{\mathbf{w}_k^T \mathbf{x} + b_k}}$$
- **Configuration:** `C=1.0`, `max_iter=1000`, `class_weight='balanced'`, `random_state=42`.
- **Scaling:** `StandardScaler()` applied to dense conversational features.
- **Strengths:** Fully interpretable coefficients, calibrated probability estimates (`predict_proba`), robust against overfitting on small sample sizes.

### Model 2: Multinomial Naive Bayes (Probabilistic Text Baseline)
- **Mathematical Principle:** Applies Bayes' theorem with the naive feature-independence assumption:
  $$P(Y = c \mid \mathbf{x}) \propto P(Y = c) \prod_{i=1}^{d} P(x_i \mid Y = c)$$
- **Configuration:** `alpha=1.0`.
- **Scaling:** `MinMaxScaler()` applied to all features to guarantee strictly non-negative inputs ($x_i \ge 0$) required by the multinomial distribution.
- **Strengths:** Rapid probabilistic inference, strong standard NLP baseline for lexical counts.

### Model 3: Linear Support Vector Machine (Margin-Based Classifier)
- **Mathematical Principle:** Finds the maximum-margin hyperplane separating classes in feature space:
  $$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_i \xi_i$$
- **Configuration:** `LinearSVC(C=1.0, class_weight='balanced', random_state=42)`.
- **Scaling:** `StandardScaler()` applied to dense conversational features.
- **Decision Score:** Computes signed margin distance $\mathbf{w}^T \mathbf{x} + b$. Calibrated via Platt scaling (`CalibratedClassifierCV`) when probability outputs are required.

---

## 3. Evaluation Strategy & Split Verification

### Clarification of Split Documentation Discrepancy
In earlier exploratory documentation, two split counts were referenced:
1. **Random Session Split (344 train / 86 test):** A naive 80/20 train/test split stratified solely on `final_understanding`. While session labels were balanced, this split allowed sessions from the same `student_id` persona to appear in both training and test partitions, creating synthetic persona leakage.
2. **Verified Group-Aware Split (357 train / 73 test):** The authoritative, leakage-free evaluation partition generated via `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped strictly on `student_id`.

### Group-Aware Validation Statistics
- **Total Dataset Size:** 430 complete sessions across 30 unique student personas (`S01` to `S30`).
- **Training Students:** 24 students (357 sessions, 83.02%).
- **Held-Out Test Students:** 6 students (73 sessions, 16.98%).
- **Student Overlap Across Train and Test:** **0 students** (100% strict group isolation).
- **Test Set Class Distribution:**
  - `understood`: 39 sessions (53.42%)
  - `partial`: 20 sessions (27.40%)
  - `confused`: 14 sessions (19.18%)
- **Leakage Elimination:** Sentiment labels (`overall_sentiment`, `turns[].sentiment`), post-session quiz scores (`quiz.score_percent`), and metadata IDs are strictly purged prior to feature pipeline ingestion.

---

## 4. Understanding Models Benchmark Comparison

All models were evaluated on the verified held-out test partition (73 sessions from 6 unseen students, 0 overlap):

| Metric | Dummy Baseline (Majority) | Logistic Regression (Balanced) | Multinomial Naive Bayes | Linear SVM (Calibrated) |
|---|---|---|---|---|
| **Test Accuracy** | 0.5342 (53.42%) | **0.9589 (95.89%)** | 0.8082 (80.82%) | **1.0000 (100%)** |
| **Macro Precision** | 0.1781 | **0.9380** | 0.7569 | **1.0000** |
| **Macro Recall** | 0.3333 | **0.9429** | 0.7167 | **1.0000** |
| **Macro F1-Score** | 0.2321 | **0.9399** | 0.7299 | **1.0000** |
| **Weighted F1-Score**| 0.3718 | **0.9591** | 0.7966 | **1.0000** |
| **5-Fold CV Accuracy**| — | **0.9378 (±0.0323)** | 0.8419 (±0.1020) | **0.9777 (±0.0182)** |
| **5-Fold CV Macro F1**| — | **0.9295 (±0.0314)** | 0.8134 (±0.0992) | **0.9744 (±0.0210)** |

### Confusion Matrix (Logistic Regression, Test Set N = 73)
```text
               Predicted
             Conf  Part  Und
True Conf     13     1     0    (Recall = 92.9%)
True Part      2    18     0    (Recall = 90.0%)
True Und       0     0    39    (Recall = 100.0%)
```

### Confusion Matrix (Linear SVM, Test Set N = 73)
```text
               Predicted
             Conf  Part  Und
True Conf     14     0     0    (Recall = 100.0%)
True Part      0    20     0    (Recall = 100.0%)
True Und       0     0    39    (Recall = 100.0%)
```

---

## 5. Feature Ablation Experiments

To quantify the exact predictive value of lexical vs. behavioral conversational signals, five feature configurations were evaluated under identical group-aware methodology (`StratifiedGroupKFold`, 357 train / 73 test, 0 student overlap):

- **Experiment A (TF-IDF Only):** 100 lexical n-gram features; all 8 conversational metrics excluded.
- **Experiment B (Conversational Signals Only):** 8 dense conversational features only; all 100 TF-IDF features excluded.
- **Experiment C (All Combined Features):** Full 108 features (8 dense signals + 100 TF-IDF).
- **Experiment D (No Explicit Understanding/Confusion Markers):** 106 features; removes explicit keyword counters (`understanding_markers`, `confusion_markers`), leaving 6 dense signals + 100 TF-IDF.
- **Experiment E (No Dense Conversational Signals):** 100 features; identical to Experiment A.

### Comparative Results Table

| Experiment Configuration | Features | LR Accuracy | LR Macro F1 | SVM Accuracy | SVM Macro F1 |
|---|---|---|---|---|---|
| **Exp A: TF-IDF Only** | 100 | 90.41% | 0.8597 | 94.52% | 0.9170 |
| **Exp B: 8 Conversational Signals Only** | 8 | 75.34% | 0.7391 | 73.97% | 0.7052 |
| **Exp C: All 108 Combined Features** | 108 | **95.89%** | **0.9399** | **100.00%** | **1.0000** |
| **Exp D: Without Explicit Markers** | 106 | 86.30% | 0.7976 | **97.26%** | **0.9620** |
| **Exp E: Without Conversational Signals** | 100 | 90.41% | 0.8597 | 94.52% | 0.9170 |
| **Exp F: No Explicit Markers & No Shortcut Vocab** | 106 | **94.52%** | **0.9265** | **98.63%** | **0.9834** |
| *Baseline: Majority Class Dummy* | *0* | *53.42%* | *0.2321* | *53.42%* | *0.2321* |

### Detailed Confusion Matrices Across Ablation Experiments

#### Exp A (TF-IDF Only, 100 feats):
- **LR:** Confused (11/14), Partial (16/20), Understood (39/39).
- **SVM:** Confused (11/14), Partial (19/20), Understood (39/39).

#### Exp B (8 Conversational Signals Only, 8 feats):
- **LR:** Confused (10/14), Partial (12/20), Understood (33/39).
- **SVM:** Confused (9/14), Partial (12/20), Understood (33/39).

#### Exp D (Without Explicit Markers, 106 feats):
- **LR:** Confused (9/14), Partial (15/20), Understood (39/39).
- **SVM:** Confused (12/14), Partial (20/20), Understood (39/39).

#### Exp F (No Explicit Markers AND No Shortcut Vocabulary in TF-IDF, 106 feats):
- **LR:** Confused (12/14), Partial (18/20), Understood (39/39) — 94.52% Acc, 0.9265 Macro F1.
- **SVM:** Confused (13/14), Partial (20/20), Understood (39/39) — 98.63% Acc, 0.9834 Macro F1 (72/73 correct).

### Pedagogical & Engineering Insights from Ablation
1. **Conversational Metrics Have Real Pedagogical Value (Exp B):** Using only 8 basic metrics (turn count, question count, word volume, etc.) achieves **75.34% accuracy** (Macro F1 = 0.7391), outperforming the 53.42% majority baseline by **+21.9%**.
2. **Empirical Evaluation of Keyword Dependence (Exp D & Exp F):**
   - In Experiment D, removing explicit counter signals (`understanding_markers`, `confusion_markers`) yields 97.26% on Linear SVM.
   - In Experiment F, removing both explicit dense counters AND all shortcut vocabulary terms (`"confused"`, `"understood"`, `"clear"`, `"lost"`, `"stuck"`, `"sorry"`, `"got"`, `"makes"`, `"sense"`, etc.) from the TF-IDF vocabulary still yields **94.52% (LR)** and **98.63% (SVM)**.
   - **Conclusion:** The model does not depend exclusively on superficial self-report phrases (e.g. "I understand" or "I'm confused"). It successfully learns from technical domain vocabulary, algorithmic reasoning expressions, and question-asking flow. However, lexical topic features still play a major role in distinguishing thorough explanations from incomplete understanding.
3. **Synergy of Lexical and Structural Features:** Combining both modalities (Exp C) delivers the best overall discrimination (**95.89%** LR, **100%** SVM).

---

## 6. Model Explainability & Feature Attributions

To guarantee transparency and pedagogical interpretability, EduLens implements model-native mathematical explainability (`ml_prototype/explainability.py`).

### A. Logistic Regression (Log-Odds Multipliers)
Since the model uses a standardized multi-class softmax formulation, the sign and magnitude of weight $w_{c,j}$ directly dictate whether feature $j$ increases or decreases the log-odds of class $c$:

| Target Class | Top Positive Features (Pushing Towards Class) | Top Negative Features (Pushing Away From Class) |
|---|---|---|
| **`confused`** | `signal_confusion_markers` (+2.84), `tfidf_read` (+1.12), `tfidf_sorry` (+1.10), `tfidf_don` (+0.81), `tfidf_handling` (+0.77) | `signal_understanding_markers` (-2.49), `tfidf_ok` (-2.27), `tfidf_clear` (-0.80), `tfidf_quiz` (-0.52) |
| **`partial`** | `signal_clarification_markers` (+1.57), `signal_question_count` (+0.95), `tfidf_wait` (+0.95), `tfidf_mean` (+0.88), `tfidf_think` (+0.75) | `signal_understanding_markers` (-1.82), `tfidf_ok` (-1.85), `tfidf_clear` (-0.68) |
| **`understood`** | `signal_understanding_markers` (+3.42), `tfidf_ok` (+3.12), `tfidf_clear` (+1.14), `tfidf_thank` (+0.93), `tfidf_perfect` (+0.89) | `signal_confusion_markers` (-2.76), `signal_clarification_markers` (-1.65), `signal_question_count` (-1.10) |

### B. Linear SVM (Decision Boundary Weights)
In Linear SVM, the weight vector $\mathbf{w}_c$ defines the normal vector to the separating hyperplane. High positive weights represent critical boundary landmarks:
- **`understood`:** Dominated by `signal_understanding_markers` (+1.87), `tfidf_ok` (+1.55), `tfidf_clear` (+0.72), `tfidf_thank` (+0.65).
- **`confused`:** Dominated by `signal_confusion_markers` (+1.52), `tfidf_sorry` (+0.64), `tfidf_read` (+0.62), `tfidf_don` (+0.54).
- **`partial`:** Characterized by elevated `signal_clarification_markers` (+0.91), `tfidf_wait` (+0.58), and moderate question frequencies.

### C. Multinomial Naive Bayes (Log-Likelihood Ratios)
Comparing class-conditional feature distributions reveals the most distinctive vocabulary terms:
- **`understood`:** `ok`, `clear`, `thank`, `perfect`, `got`, `makes sense`, `solve`.
- **`confused`:** `sorry`, `don`, `stuck`, `lost`, `confused`, `handling`, `pointer`.
- **`partial`:** `wait`, `mean`, `example`, `why`, `step`, `difference`.

### D. Local Instance Feature Attribution (`explain_instance`)
When predicting on a single conversation, EduLens computes local feature attribution:
$$\text{attribution}_j = w_{\hat{c}, j} \cdot x_j$$
Where $x_j$ is the transformed feature value and $w_{\hat{c}, j}$ is the model weight for the predicted class. The dashboard and API expose positive drivers (evidence supporting the verdict) and negative drivers (opposing evidence).

---

## 7. Secondary Model: Quiz Score Regressor

### Problem Formulation & Group-Aware Split
Predicts post-session student assessment score percentage ($y \in [0, 100]$) from conversation features.
- **Target:** `quiz.score_percent`
- **Dataset:** 1,720 sessions (`edulens_clean_quiz.json`: 430 primary + 1,290 supplementary sets).
- **Split Protocol:** `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` grouped strictly on `student_id`.
  - **Training:** 1,394 sessions across 32 unique students.
  - **Testing:** 326 sessions across 8 unique students.
  - **Student Overlap:** **0 students** (strict group independence).

### Group-Aware Regression Benchmarks

| Model Architecture | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | $R^2$ Score |
|---|---|---|---|
| **Ridge Regressor (L2, $\alpha=1.0$)** | 7.76% | 9.68% | 0.7460 |
| **Random Forest Regressor (100 trees)** | **7.65%** | **9.36%** | **0.7624** |

### Key Regression Drivers
Feature importance analysis reveals the primary drivers of predicted quiz performance:
1. `signal_understanding_markers`: Strongest positive driver (RF Gini importance: 0.5406, Ridge weight: +6.82).
2. `tfidf_ok`: Secondary positive confirmation driver (RF importance: 0.1175).
3. `signal_total_words` & `signal_turn_length`: Moderate positive correlation with score (+1.45 Ridge).
4. `signal_confusion_markers`: Strongest negative driver (-3.85 Ridge).
5. `signal_question_count`: Slight negative coefficient (-0.88 Ridge) reflecting unresolved circular queries.

---

## 8. Critical Evaluation Limitations

1. **Synthetic Ground Truth:** The labels in `edulens_chats_dataset.json` were synthesized using Claude with structured pedagogical persona prompts. Evaluation measures model agreement with these synthetic personas, not proven human psychometric understanding.
2. **Domain Focus:** Evaluated on Computer Science university topics. Generalization to humanities, clinical, or qualitative domains remains unverified.
3. **Instructional Support Tool:** Predictions are intended to assist teachers and students in flagging difficulty hotspots, never to replace formal university grading or certification.
