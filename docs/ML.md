# Machine Learning (ML) — EduLens

> **Status:** Active Reference & Permanent Technical Record  
> **Module Responsibility:** Model Architecture, Training, Cross-Validation, Evaluation Benchmarks, Inference  
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
- **Strengths:** Fully interpretable coefficients, calibrated probability estimates (`predict_proba`), robust against overfitting on small sample sizes.

### Model 2: Multinomial Naive Bayes (Probabilistic Text Baseline)
- **Mathematical Principle:** Applies Bayes' theorem with the naive feature-independence assumption:
  $$P(Y = c \mid \mathbf{x}) \propto P(Y = c) \prod_{i=1}^{d} P(x_i \mid Y = c)$$
- **Usage:** Serves as a standard NLP benchmark for text and term-count classifications. Dense features are shifted/min-max scaled to ensure non-negativity.

### Model 3: Linear Support Vector Machine (Margin-Based Classifier)
- **Mathematical Principle:** Finds the maximum-margin hyperplane separating classes in feature space:
  $$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_i \xi_i$$
- **Configuration:** `LinearSVC(C=1.0, class_weight='balanced', random_state=42)`.
- **Strengths:** High dimensional margin optimization, effective when class boundaries are cleanly separable. Calibrated via Platt scaling (`CalibratedClassifierCV`) if probabilities are required.

---

## 3. Evaluation Strategy & Leakage Prevention

### Validation Protocol
1. **No Target Leakage:** Sentiment labels and post-session quiz results are strictly excluded from input features.
2. **Train/Test Splitting:**
   - 80% Training (344 sessions)
   - 20% Held-Out Testing (86 sessions)
   - Group-aware validation using `student_id` prevents persona leakage across splits.
3. **Cross-Validation:** 5-fold Stratified / GroupKFold cross-validation on the training set.
4. **Primary Comparison Metrics:**
   - **Macro F1-Score:** Unweighted average of F1 across classes, essential due to class imbalance (`confused` = 19%).
   - **Accuracy, Macro Precision, Macro Recall.**
   - **Full Confusion Matrix.**

---

## 4. Understanding Models Benchmark Comparison

All models were evaluated on the held-out test partition (73 sessions from 6 unseen students) with zero student overlap using `StratifiedGroupKFold(n_splits=5, student_id)`.

| Metric | Dummy Baseline (Majority) | Logistic Regression (Balanced) | Multinomial Naive Bayes | Linear SVM (Calibrated) |
|---|---|---|---|---|
| **Test Accuracy** | 0.5342 (53.42%) | **0.9589 (95.89%)** | 0.8082 (80.82%) | **1.0000 (100%)** |
| **Macro Precision** | — | **0.9380** | 0.7569 | **1.0000** |
| **Macro Recall** | — | **0.9429** | 0.7167 | **1.0000** |
| **Macro F1-Score** | 0.2321 | **0.9399** | 0.7299 | **1.0000** |
| **Weighted F1-Score**| — | **0.9591** | 0.7966 | **1.0000** |
| **5-Fold CV Accuracy**| — | **0.9378 (±0.0323)** | 0.8419 (±0.1020) | **0.9777 (±0.0182)** |
| **5-Fold CV Macro F1**| — | **0.9295 (±0.0314)** | 0.8134 (±0.0992) | **0.9744 (±0.0210)** |

### Confusion Matrix (Logistic Regression, Test Set N = 73)
- **Confused (True = 14):** 13 correct, 1 predicted Partial, 0 predicted Understood.
- **Partial (True = 20):** 18 correct, 2 predicted Confused, 0 predicted Understood.
- **Understood (True = 39):** 39 correct, 0 misclassified.

### Confusion Matrix (Linear SVM, Test Set N = 73)
- **Confused (True = 14):** 14 correct (100%).
- **Partial (True = 20):** 20 correct (100%).
- **Understood (True = 39):** 39 correct (100%).

---

## 5. Additional Models: Quiz Score Regressor

### Problem Formulation
Predicts post-session student assessment score percentage ($y \in [0, 100]$) from conversation features.
- **Target:** `quiz.score_percent`
- **Dataset:** 1,720 sessions (`edulens_clean_quiz.json`: 430 primary + 1,290 supplementary sets).
- **Split:** 80% Train (1,376 sessions), 20% Test (344 sessions).

### Regression Benchmarks

| Model Architecture | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | $R^2$ Score |
|---|---|---|---|
| **Ridge Regressor (L2)** | 8.013% | 9.856 | 0.7356 |
| **Random Forest Regressor** | **8.006%** | **9.817** | **0.7377** |

---

## 6. Critical Evaluation Limitations

1. **Synthetic Ground Truth:** The labels in `edulens_chats_dataset.json` were synthesized using Claude with structured pedagogical persona prompts. Evaluation measures agreement with these synthetic personas, not proven human psychometric understanding.
2. **Domain Focus:** Evaluated on Computer Science university topics. Generalization to humanities, clinical, or qualitative domains remains unverified.
3. **Instructional Support Tool:** Predictions are intended to assist teachers and students in flagging difficulty hotspots, never to replace formal university grading or certification.
