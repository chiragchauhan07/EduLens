"""
dashboard.py — Streamlit Interactive Web Application for EduLens ML Prototype.

Three Focused Pages:
1. Student Performance Analyzer (Conversation analysis & real-time ML prediction)
2. Topic-Wise Performance (Curriculum-level topic aggregation and confusion analysis)
3. Model Information & Viva Guide (Transparent metrics, baseline comparison, and limitations)
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml_prototype.data_loader import extract_student_messages, load_primary_dataset
from ml_prototype.predict import EduLensPredictor, CLASS_METADATA


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION & STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduLens — Learning Intelligence Prototype",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-understood {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-partial {
        background-color: #FEF08A;
        color: #713F12;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-confused {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .chat-bubble {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor(algorithm: str = "logistic_regression"):
    """Cache the loaded ML model pipeline for the given algorithm."""
    return EduLensPredictor(algorithm=algorithm)


@st.cache_data
def get_dataset():
    """Cache the loaded primary dataset."""
    return load_primary_dataset()


@st.cache_data
def get_model_evaluation():
    """Load model evaluation JSON report."""
    report_path = ROOT_DIR / "reports" / "model_evaluation.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def get_model_comparison():
    """Load cross-model comparison JSON report."""
    report_path = ROOT_DIR / "reports" / "model_comparison.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def get_quiz_evaluation():
    """Load quiz score regressor evaluation report."""
    report_path = ROOT_DIR / "reports" / "quiz_regressor_evaluation.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def get_feature_ablation():
    """Load feature ablation JSON report."""
    report_path = ROOT_DIR / "reports" / "feature_ablation.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def get_model_explainability():
    """Load model explainability JSON report."""
    report_path = ROOT_DIR / "reports" / "model_explainability.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
st.sidebar.title("EduLens ML")
st.sidebar.caption("Student Learning Intelligence Prototype")

page = st.sidebar.radio(
    "Navigation",
    ["Student Performance Analyzer", "Topic-Wise Performance", "Model Information"],
    index=0,
)

st.sidebar.divider()
st.sidebar.info(
    "**Core Principle:**\n\n"
    "EduLens does not tutor students. It analyzes student-LLM conversations to estimate "
    "learning understanding through conversation-derived NLP evidence."
)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: STUDENT PERFORMANCE ANALYZER
# ─────────────────────────────────────────────────────────────────────────────
if page == "Student Performance Analyzer":
    st.markdown('<div class="main-title">Student Performance Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Evaluate a student-LLM learning conversation to predict '
        'demonstrated understanding level using trained machine learning.</div>',
        unsafe_allow_html=True,
    )

    col_top1, col_top2 = st.columns([2, 1])
    with col_top1:
        algo_choice = st.selectbox(
            "Select Understanding Classifier Algorithm:",
            options=[
                ("logistic_regression", "Logistic Regression (Primary — Calibrated Probabilities)"),
                ("linear_svm", "Linear Support Vector Machine (Benchmark — Max-Margin)"),
                ("naive_bayes", "Multinomial Naive Bayes (Benchmark — Generative)"),
            ],
            format_func=lambda x: x[1],
            index=0,
        )
        selected_algo = algo_choice[0]
    with col_top2:
        st.write("")
        st.write("")
        enable_explain = st.checkbox("Enable Feature Attribution (Explainability)", value=True)

    try:
        predictor = get_predictor(selected_algo)
        dataset_df = get_dataset()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

    input_mode = st.radio(
        "Select Input Source:",
        ["Curated Sample Conversations", "Upload JSON File", "Paste Custom Conversation Text"],
        horizontal=True,
    )

    current_session = {}

    if input_mode == "Curated Sample Conversations":
        # Provide sample index
        samples = [
            {"label": "Sample 1 (C++ Pointers — Struggling/Confused)", "index": 0},
            {"label": "Sample 2 (DAA Binary Search — Mastered/Understood)", "index": 8},
            {"label": "Sample 3 (OS Virtual Memory — Emerging/Partial)", "index": 25},
            {"label": "Sample 4 (DSA Graph BFS — Mastered/Understood)", "index": 50},
            {"label": "Sample 5 (ML Gradient Descent — Confused)", "index": 95},
        ]
        sample_choice = st.selectbox(
            "Choose a sample conversation from verified dataset:",
            options=range(len(samples)),
            format_func=lambda i: samples[i]["label"],
        )
        chosen_idx = samples[sample_choice]["index"]
        if chosen_idx < len(dataset_df):
            raw_rec = dataset_df.iloc[chosen_idx].to_dict()
            current_session = {
                "chat_id": raw_rec.get("chat_id"),
                "subject": raw_rec.get("subject"),
                "topic": raw_rec.get("topic"),
                "student_persona": raw_rec.get("student_persona"),
                "student_messages": raw_rec.get("student_messages", []),
                "actual_ground_truth": raw_rec.get("final_understanding"),
            }

    elif input_mode == "Upload JSON File":
        uploaded = st.file_uploader("Upload a JSON conversation session:", type=["json"])
        if uploaded:
            try:
                data = json.load(uploaded)
                if isinstance(data, list) and len(data) > 0:
                    data = data[0]
                msgs = extract_student_messages(data)
                current_session = {
                    "chat_id": data.get("chat_id", "UPLOADED_SESSION"),
                    "subject": data.get("subject", "Uploaded Subject"),
                    "topic": data.get("topic", "Uploaded Topic"),
                    "student_persona": data.get("student_persona", "custom"),
                    "student_messages": msgs,
                    "actual_ground_truth": data.get("final_understanding", None),
                }
            except Exception as e:
                st.error(f"Error reading JSON: {e}")

    else:
        # Paste Custom Text
        st.caption("Enter student utterances (one message per line):")
        custom_text = st.text_area(
            "Student messages:",
            height=150,
            value=(
                "Can you explain how quicksort partition works?\n"
                "I'm a bit confused about why we pick the last element as pivot.\n"
                "Oh wait, so elements smaller go left and larger go right? Got it, that makes sense now!"
            ),
        )
        custom_subject = st.text_input("Subject (Optional):", value="Algorithms")
        custom_topic = st.text_input("Topic (Optional):", value="Quicksort Partitioning")
        lines = [l.strip() for l in custom_text.split("\n") if l.strip()]
        current_session = {
            "chat_id": "CUSTOM_INPUT",
            "subject": custom_subject,
            "topic": custom_topic,
            "student_persona": "Interactive User",
            "student_messages": lines,
            "actual_ground_truth": None,
        }

    # Display Conversation Preview
    if current_session and current_session.get("student_messages"):
        st.subheader("Conversation Metadata & Messages")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Chat ID", current_session.get("chat_id", "N/A"))
        col_m2.metric("Subject", current_session.get("subject", "N/A"))
        col_m3.metric("Topic", current_session.get("topic", "N/A"))
        col_m4.metric("Student Persona", str(current_session.get("student_persona", "N/A")).capitalize())

        with st.expander("Inspect Student Message Transcript", expanded=True):
            for i, msg in enumerate(current_session["student_messages"], start=1):
                st.markdown(
                    f'<div class="chat-bubble"><b>Turn {i} (Student):</b> {msg}</div>',
                    unsafe_allow_html=True,
                )

        st.write("")
        analyze_clicked = st.button("🔍 Analyze Learning Session", type="primary", use_container_width=True)

        if analyze_clicked:
            with st.spinner("Extracting NLP features and running model inference..."):
                result = predictor.predict(current_session, explain=enable_explain)

            st.write("---")
            st.subheader("Analysis & ML Prediction Results")

            pred_class = result["predicted_class"]
            badge_class = f"badge-{pred_class}"

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown("#### Predicted Learning Outcome")
                st.markdown(
                    f'<span class="{badge_class}" style="font-size:1.25rem;">{result["display_name"]}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Inference Model: **{selected_algo.replace('_', ' ').title()}**")
                st.write("")
                col_c1, col_c2 = st.columns(2)
                col_c1.metric("Model Confidence", f"{result['confidence']:.1%}")
                if result.get("predicted_quiz_score") is not None:
                    col_c2.metric("Predicted Quiz Score", f"{result['predicted_quiz_score']:.1f}%")

                if current_session.get("actual_ground_truth"):
                    gt = current_session["actual_ground_truth"]
                    match_str = "✅ Match" if gt == pred_class else "⚠️ Mismatch"
                    st.caption(f"Dataset Ground Truth: **{gt}** ({match_str})")

                st.markdown("##### Pedagogical Recommendation:")
                st.info(result["pedagogical_feedback"])

            with res_col2:
                if selected_algo == "linear_svm" and result.get("feature_explanation", {}).get("decision_scores"):
                    st.markdown("#### Margin Decision Scores (Linear SVM)")
                    st.caption("Signed distance to one-vs-rest separating hyperplanes (higher = stronger class association):")
                    for cls, score in result["feature_explanation"]["decision_scores"].items():
                        st.write(f"**{cls.capitalize()}**: `{score:+.3f}`")
                else:
                    st.markdown("#### Model Class Probabilities")
                    for cls, prob in result["probabilities"].items():
                        st.write(f"**{cls.capitalize()}**: `{prob:.1%}`")
                        st.progress(float(prob))

            st.write("")
            st.markdown("#### Extracted Conversational Evidence Signals")
            sig = result["conversational_signals"]
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Student Turns", int(sig.get("num_turns", 0)))
            sc2.metric("Total Words", int(sig.get("total_words", 0)))
            sc3.metric("Avg Words / Turn", f"{sig.get('avg_turn_length', 0):.1f}")
            sc4.metric("Questions Asked", int(sig.get("question_count", 0)))

            sc5, sc6, sc7, sc8 = st.columns(4)
            sc5.metric("Confusion Signals", int(sig.get("confusion_markers", 0)))
            sc6.metric("Clarification Requests", int(sig.get("clarification_markers", 0)))
            sc7.metric("Comprehension Signals", int(sig.get("understanding_markers", 0)))
            sc8.metric("Technical/Code Tokens", int(sig.get("code_indicators", 0)))

            # Feature Attribution / Local Explainability
            if enable_explain and result.get("feature_explanation"):
                exp_data = result["feature_explanation"]
                st.write("")
                st.subheader("💡 Model Explainability & Local Evidence Attribution")
                if "summary" in exp_data:
                    st.info(f"**Attribution Summary:** {exp_data['summary']}")

                exp_col1, exp_col2 = st.columns(2)
                with exp_col1:
                    st.markdown(f"##### 🟢 Evidence Supporting *'{pred_class.upper()}'*")
                    pos_feats = exp_data.get("top_positive_features", [])
                    if pos_feats:
                        pos_df = pd.DataFrame([{
                            "Feature": item["feature"].replace("signal_", "📊 ").replace("tfidf_", "🔤 '") + ("'" if "tfidf_" in item["feature"] else ""),
                            "Observed Value": f"{item['value']:.2f}" if isinstance(item['value'], float) else str(item['value']),
                            "Attribution Weight": f"{item['weight']:+.3f}",
                        } for item in pos_feats])
                        st.dataframe(pos_df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No strong positive features identified.")

                with exp_col2:
                    st.markdown(f"##### 🔴 Opposing Evidence / Contrasting Signals")
                    neg_feats = exp_data.get("top_negative_features", [])
                    if neg_feats:
                        neg_df = pd.DataFrame([{
                            "Feature": item["feature"].replace("signal_", "📊 ").replace("tfidf_", "🔤 '") + ("'" if "tfidf_" in item["feature"] else ""),
                            "Observed Value": f"{item['value']:.2f}" if isinstance(item['value'], float) else str(item['value']),
                            "Attribution Weight": f"{item['weight']:+.3f}",
                        } for item in neg_feats])
                        st.dataframe(neg_df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No strong opposing features identified.")

    else:
        st.warning("Please provide a conversation containing student messages to analyze.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: TOPIC-WISE PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Topic-Wise Performance":
    st.markdown('<div class="main-title">Topic-Wise Performance Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Curriculum-level aggregation of student understanding and '
        'confusion patterns across verified academic topics.</div>',
        unsafe_allow_html=True,
    )

    try:
        dataset_df = get_dataset()
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        st.stop()

    # Aggregate by topic
    topic_summary = []
    for topic_name, group in dataset_df.groupby("topic"):
        n_sessions = len(group)
        counts = group["final_understanding"].value_counts()
        understood_pct = round((counts.get("understood", 0) / n_sessions) * 100, 1)
        partial_pct = round((counts.get("partial", 0) / n_sessions) * 100, 1)
        confused_pct = round((counts.get("confused", 0) / n_sessions) * 100, 1)
        avg_turns = round(group["num_turns"].mean(), 1)
        subject = group["subject"].iloc[0]

        topic_summary.append({
            "Topic": topic_name,
            "Subject": subject,
            "Total Sessions": n_sessions,
            "% Understood": understood_pct,
            "% Partial": partial_pct,
            "% Confused": confused_pct,
            "Avg Turns": avg_turns,
        })

    summary_df = pd.DataFrame(topic_summary).sort_values(by="Total Sessions", ascending=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Unique Topics", len(summary_df))
    col2.metric("Total Sessions Analyzed", len(dataset_df))
    avg_mastery = round(summary_df["% Understood"].mean(), 1)
    col3.metric("Curriculum Average Mastery", f"{avg_mastery}%")

    st.subheader("Topic Aggregation Table")
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
    )

    st.write("---")
    st.subheader("Topic Drill-Down")
    selected_topic = st.selectbox(
        "Select a topic to inspect session distribution and confusion hotspots:",
        options=summary_df["Topic"].tolist(),
    )

    topic_sessions = dataset_df[dataset_df["topic"] == selected_topic]
    d_col1, d_col2 = st.columns([1, 1])

    with d_col1:
        st.markdown(f"#### Distribution for: *{selected_topic}*")
        dist = topic_sessions["final_understanding"].value_counts()
        for k in ["understood", "partial", "confused"]:
            cnt = dist.get(k, 0)
            pct = (cnt / len(topic_sessions)) * 100
            st.write(f"**{k.capitalize()}**: {cnt} sessions ({pct:.1f}%)")
            st.progress(pct / 100.0)

    with d_col2:
        st.markdown("#### Individual Sessions in this Topic")
        for idx, row in topic_sessions.iterrows():
            with st.expander(f"{row['chat_id']} — Outcome: {row['final_understanding'].upper()} ({row['num_turns']} turns)"):
                st.caption(f"Student: {row['student_id']} | Persona: {row['student_persona']}")
                st.write(row["full_text"][:200] + "...")

    st.write("---")
    st.info(
        "**Methodological Note on Topic Aggregation:**\n\n"
        "The sessions shown above represent distinct simulated learning interactions from the verified dataset. "
        "They illustrate topic difficulty patterns across learners. Real student longitudinal growth tracking requires "
        "persisted student user IDs across multiple learning dates."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: MODEL INFORMATION & VIVA GUIDE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model Information":
    st.markdown('<div class="main-title">Model Architecture & Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Transparent technical documentation, baseline comparison, '
        'and viva explanation reference.</div>',
        unsafe_allow_html=True,
    )

    eval_report = get_model_evaluation()

    if not eval_report:
        st.warning("Model evaluation report not found. Run `python ml_prototype/train_model.py` to generate it.")
        st.stop()

    # Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Architecture", eval_report.get("model_type", "LogisticRegression"))
    c2.metric("Target Variable", "final_understanding")
    c3.metric("Test Accuracy", f"{eval_report['evaluation_metrics']['test_accuracy']:.1%}")
    c4.metric("Majority Baseline", f"{eval_report['baseline_metrics']['accuracy']:.1%}")

    st.write("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("1. Problem Formulation")
        st.write(
            "EduLens formulates learning outcome prediction as a **supervised 3-class classification** task:\n"
            "- **Input:** Student utterances extracted from a learning conversation with an LLM.\n"
            "- **Output:** Predicted understanding level (`understood`, `partial`, `confused`).\n"
            "- **Target Selection Rationale:** Categorical understanding directly represents Bloom's cognitive progression "
            "without imposing arbitrary numerical conversions or fabricated quiz formulas."
        )

        st.subheader("2. Feature Engineering Pipeline")
        st.write(
            "The model relies on a leakage-free Scikit-Learn `ColumnTransformer`:\n"
            "1. **TF-IDF Vocabulary Vectorizer (100 features):** Unigrams and bigrams capturing conceptual mastery vs. hesitation terms.\n"
            "2. **Dense Conversational Signals (8 features):** Scaled via `StandardScaler`:\n"
            "   - Turn count & total words\n"
            "   - Average words per turn\n"
            "   - Question marks count\n"
            "   - Confusion phrase markers\n"
            "   - Clarification request markers\n"
            "   - Understanding affirmation markers\n"
            "   - Technical / code syntax indicators\n"
            "- **Leakage Prevention:** Does not use sentiment or post-session quiz data."
        )

    with col_r:
        st.subheader("3. Rigorous Evaluation Results")
        m = eval_report["evaluation_metrics"]
        st.table(pd.DataFrame({
            "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score", "Weighted F1-Score"],
            "Primary Model (Logistic Reg)": [
                f"{m['test_accuracy']:.4f}",
                f"{m['test_macro_precision']:.4f}",
                f"{m['test_macro_recall']:.4f}",
                f"{m['test_macro_f1']:.4f}",
                f"{m['test_weighted_f1']:.4f}",
            ],
            "Baseline (Dummy Majority)": [
                f"{eval_report['baseline_metrics']['accuracy']:.4f}",
                "—",
                "—",
                f"{eval_report['baseline_metrics']['macro_f1']:.4f}",
                "—",
            ],
        }))

        st.subheader("4. 5-Fold Stratified Group Cross-Validation")
        st.write(
            f"- **Mean CV Accuracy:** `{m['cv_5fold_accuracy_mean']:.4f} (+/- {m['cv_5fold_accuracy_std']:.4f})`\n"
            f"- **Mean CV Macro F1:** `{m['cv_5fold_f1_macro_mean']:.4f} (+/- {m['cv_5fold_f1_macro_std']:.4f})`\n"
            f"- **Split Strategy:** Group-aware splitting on `student_id` (Zero persona overlap across train and test)."
        )

    # Cross-Model Benchmark Comparison Section
    comp_report = get_model_comparison()
    if comp_report and "models" in comp_report:
        st.write("---")
        st.subheader("5. Multi-Model Benchmark Comparison")
        st.caption("Comparison across Logistic Regression, Naive Bayes, and Linear Support Vector Machine on held-out test students:")
        
        comp_rows = []
        for m_key, m_data in comp_report["models"].items():
            comp_rows.append({
                "Algorithm": m_data["display_name"],
                "Test Accuracy": f"{m_data['test_accuracy']:.4f}",
                "Macro F1": f"{m_data['macro_f1']:.4f}",
                "Macro Recall": f"{m_data['macro_recall']:.4f}",
                "5-Fold CV Accuracy": f"{m_data['cv_accuracy_mean']:.4f} (±{m_data['cv_accuracy_std']:.4f})",
            })
        st.table(pd.DataFrame(comp_rows))

    # Feature Ablation Section
    ablation_report = get_feature_ablation()
    if ablation_report and "experiments" in ablation_report:
        st.write("---")
        st.subheader("6. Feature Ablation Experiments (6 Configurations)")
        st.caption(
            "Systematic evaluation across 6 feature subsets to measure the exact predictive contribution of lexical "
            "vs. behavioral conversational features under identical Stratified Group splitting (357 train / 73 test, 0 student overlap):"
        )

        ablation_rows = []
        for exp_key, exp_data in ablation_report["experiments"].items():
            lr_res = exp_data.get("logistic_regression", {})
            svm_res = exp_data.get("linear_svm", {})
            ablation_rows.append({
                "Experiment": exp_data["name"],
                "Features": exp_data["feature_count"],
                "LR Accuracy": f"{lr_res.get('accuracy', 0):.2%}",
                "LR Macro F1": f"{lr_res.get('macro_f1', 0):.4f}",
                "SVM Accuracy": f"{svm_res.get('accuracy', 0):.2%}",
                "SVM Macro F1": f"{svm_res.get('macro_f1', 0):.4f}",
            })
        st.table(pd.DataFrame(ablation_rows))

        st.info(
            "**Key Ablation Findings:**\n\n"
            "- **Exp B (8 Dense Conversational Signals Only):** Achieves **75.3% accuracy** (Macro F1 = 0.7391), outperforming "
            "the 53.4% dummy baseline by **+21.9%**. This confirms that conversation flow metrics carry genuine pedagogical signal.\n"
            "- **Exp C (All 108 Combined Features):** Yields the strongest overall performance (**95.9% LR, 100% SVM**).\n"
            "- **Exp D (No Explicit Understanding/Confusion Markers):** Achieves **97.3% accuracy / 0.9620 Macro F1** on Linear SVM.\n"
            "- **Exp F (No Explicit Markers AND No Shortcut Vocabulary in TF-IDF):** Even when explicit markers and shortcut terms "
            "('confused', 'understood', 'clear', 'lost', 'stuck', etc.) are purged from BOTH dense signals and TF-IDF vocabulary, "
            "the model achieves **94.5% (LR)** and **98.6% (SVM)** accuracy, demonstrating that the classifier learns genuine "
            "dialogue reasoning and contextual inquiry rather than memorizing superficial keywords."
        )

    # Model Explainability & Feature Drivers Section
    explain_data = get_model_explainability()
    if explain_data:
        st.write("---")
        st.subheader("7. Global Model Explainability & Key Feature Drivers")
        st.caption("Interpreting the mathematical parameters that govern class predictions and regression scores:")

        tab_lr, tab_svm, tab_nb, tab_quiz = st.tabs([
            "Logistic Regression (Coefficients)",
            "Linear SVM (Margin Weights)",
            "Multinomial Naive Bayes (Log-Odds)",
            "Quiz Score Regressor (Feature Importance)",
        ])

        with tab_lr:
            st.markdown("##### Logistic Regression Class Drivers (Top Log-Odds Multipliers)")
            lr_exps = explain_data.get("logistic_regression", {}).get("class_explanations", {})
            cls_sel = st.selectbox("Select Target Class:", ["understood", "partial", "confused"], key="exp_lr_cls")
            if cls_sel in lr_exps:
                c_data = lr_exps[cls_sel]
                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    st.markdown(f"**Top Positive Features (Pushing Towards '{cls_sel}')**")
                    pos_df = pd.DataFrame(c_data.get("top_positive_features", [])[:8])
                    st.dataframe(pos_df, use_container_width=True, hide_index=True)
                with c_col2:
                    st.markdown(f"**Top Negative Features (Pushing Away from '{cls_sel}')**")
                    neg_df = pd.DataFrame(c_data.get("top_negative_features", [])[:8])
                    st.dataframe(neg_df, use_container_width=True, hide_index=True)

        with tab_svm:
            st.markdown("##### Linear SVM Decision Boundary Weights")
            svm_exps = explain_data.get("linear_svm", {}).get("class_explanations", {})
            svm_cls_sel = st.selectbox("Select Target Class:", ["understood", "partial", "confused"], key="exp_svm_cls")
            if svm_cls_sel in svm_exps:
                s_data = svm_exps[svm_cls_sel]
                s_col1, s_col2 = st.columns(2)
                with s_col1:
                    st.markdown(f"**Highest Positive Margin Weights ('{svm_cls_sel}')**")
                    pos_df = pd.DataFrame(s_data.get("top_positive_features", [])[:8])
                    st.dataframe(pos_df, use_container_width=True, hide_index=True)
                with s_col2:
                    st.markdown(f"**Highest Negative Margin Weights ('{svm_cls_sel}')**")
                    neg_df = pd.DataFrame(s_data.get("top_negative_features", [])[:8])
                    st.dataframe(neg_df, use_container_width=True, hide_index=True)

        with tab_nb:
            st.markdown("##### Multinomial Naive Bayes Discriminative Terms")
            nb_exps = explain_data.get("naive_bayes", {}).get("class_explanations", {})
            nb_cls_sel = st.selectbox("Select Target Class:", ["understood", "partial", "confused"], key="exp_nb_cls")
            if nb_cls_sel in nb_exps:
                n_data = nb_exps[nb_cls_sel]
                st.markdown(f"**Most Discriminative Vocabulary Terms for '{nb_cls_sel}' (Log-Likelihood Ratio vs Other Classes)**")
                n_df = pd.DataFrame(n_data.get("top_distinctive_terms", [])[:10])
                st.dataframe(n_df, use_container_width=True, hide_index=True)

        with tab_quiz:
            st.markdown("##### Quiz Score Regressor Drivers")
            q_data = explain_data.get("quiz_regressor", {})
            q_col1, q_col2 = st.columns(2)
            with q_col1:
                st.markdown("**Ridge Regressor Coefficients (Feature Weights)**")
                r_weights = q_data.get("ridge_coefficients", [])
                if r_weights:
                    r_df = pd.DataFrame(r_weights[:10])
                    st.dataframe(r_df, use_container_width=True, hide_index=True)
            with q_col2:
                st.markdown("**Random Forest Gini Feature Importances**")
                rf_importances = q_data.get("random_forest_importance", [])
                if rf_importances:
                    rf_df = pd.DataFrame(rf_importances[:10])
                    st.dataframe(rf_df, use_container_width=True, hide_index=True)

    # Quiz Score Regressor Section
    quiz_eval = get_quiz_evaluation()
    if quiz_eval and "models" in quiz_eval:
        st.write("---")
        st.subheader("8. Secondary Model: Quiz Score Regressor")
        st.caption(f"Trained on {quiz_eval['dataset_records']} student sessions with post-session assessment scores:")
        
        quiz_rows = []
        for q_key, q_data in quiz_eval["models"].items():
            quiz_rows.append({
                "Regression Model": q_data["name"],
                "Mean Absolute Error (MAE)": f"{q_data['mae']:.2f}%",
                "Root Mean Squared Error (RMSE)": f"{q_data['rmse']:.2f}%",
                "R² Score": f"{q_data['r2']:.4f}",
            })
        st.table(pd.DataFrame(quiz_rows))

    st.write("---")
    st.subheader("9. Important Honest Limitations")
    for lim in eval_report.get("limitations", []):
        st.markdown(f"- ⚠️ {lim}")
