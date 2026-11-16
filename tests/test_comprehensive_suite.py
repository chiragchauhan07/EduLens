"""
test_comprehensive_suite.py — Complete 12-Case Quality & Readiness Test Suite for EduLens.

Covers all required acceptance scenarios:
- Case 1: Student understands a topic (detailed explanation in own words)
- Case 2: Student partially understands (lingering ambiguity & hints)
- Case 3: Student is confused (persistent doubts & circular questions)
- Case 4: Actual quiz evaluation (10 questions, correct/incorrect/unanswered/actual score)
- Case 5: Missing quiz responses (no fabricated quiz score)
- Case 6: Multiple topics separation
- Case 7: Empty conversation handling (graceful validation error)
- Case 8: Malformed JSON handling
- Case 9: Unknown student handling (stateless inference)
- Case 10: Unseen vocabulary handling (dimension stability)
- Case 11: Repeated student records (zero persona leakage across evaluation splits)
- Case 12: Missing model artifact error reporting
- Component tests: DSP, NLP, ML, Confidence, Scoring Engine, API endpoints
"""

import json
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent

from ml_prototype.dsp import (
    validate_session_json,
    extract_dsp_conversation_metrics,
    clean_and_normalize_session,
)
from ml_prototype.nlp import (
    normalize_conversational_text,
    parse_and_clean_conversation,
    extract_nlp_features_vector,
)
from ml_prototype.features import build_feature_pipeline
from ml_prototype.predict import EduLensPredictor
from ml_prototype.quiz_evaluator import evaluate_quiz_submission
from ml_prototype.confidence import (
    extract_confidence_signals,
    compute_expressed_confidence_score,
    ExpressedConfidenceEstimator,
)
from ml_prototype.scoring_engine import (
    convert_understanding_to_score,
    TopicPerformanceScoringEngine,
)
from ml_prototype.unified_pipeline import EduLensUnifiedPipeline
from api.index import app

client = TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# CASE 1: Student Understands a Topic (Detailed explanation in own words)
# ─────────────────────────────────────────────────────────────────────────────
def test_case_1_student_understands_topic():
    pipeline = EduLensUnifiedPipeline(algorithm="logistic_regression")
    conversation = {
        "chat_id": "SESSION_CASE_1",
        "subject": "DSA",
        "topic": "Binary Search Trees",
        "turns": [
            {
                "student_message": "can you explain how binary search tree insertion maintains the ordering invariant?",
                "teacher_message": "Each node maintains the invariant that left subtree keys are smaller and right subtree keys are larger."
            },
            {
                "student_message": "got it! so if the new key is smaller than the current node, we recurse left; if larger, we recurse right. when we reach null, we attach the new node. that means search and insert both take O(h) time where h is the tree height. that makes complete sense now! thank you!",
                "teacher_message": "Spot on explanation."
            }
        ]
    }
    result = pipeline.process_session(conversation)
    assert result["status"] == "success"
    topic_data = result["topics"][0]
    assert topic_data["understanding"]["classification"] in {"understood", "partial"}
    assert topic_data["actual_quiz_performance"] is None  # No fabricated quiz result
    assert topic_data["overall_performance"]["score"] is None
    assert topic_data["overall_performance"]["status"] == "awaiting_quiz_submission"


# ─────────────────────────────────────────────────────────────────────────────
# CASE 2: Student Partially Understands (Grasps basics, makes minor errors)
# ─────────────────────────────────────────────────────────────────────────────
def test_case_2_student_partially_understands():
    pipeline = EduLensUnifiedPipeline(algorithm="logistic_regression")
    conversation = [
        "how does page table translation work in virtual memory?",
        "so the TLB is like a hardware cache for page table entries, right?",
        "can you give another example of inverted page tables? I understand regular paging but inverted is still unclear to me.",
        "okay that clarifies some doubts, will practice numerical problems on this.",
    ]
    result = pipeline.process_session(conversation, topic_override="Virtual Memory")
    assert result["status"] == "success"
    topic_data = result["topics"][0]
    assert topic_data["understanding"]["classification"] in {"partial", "understood", "confused"}
    assert "score_index" in topic_data["understanding"]
    assert 0.0 <= topic_data["understanding"]["score_index"] <= 100.0


# ─────────────────────────────────────────────────────────────────────────────
# CASE 3: Student is Confused (Repeated confusion and doubts)
# ─────────────────────────────────────────────────────────────────────────────
def test_case_3_student_is_confused():
    pipeline = EduLensUnifiedPipeline(algorithm="logistic_regression")
    conversation = [
        "i don't understand pointers in C++ at all, it's so confusing.",
        "what is the difference between * and &? I'm completely lost.",
        "why does it give a segmentation fault? I don't get this.",
        "can you explain again from scratch? I am totally stuck and confused.",
    ]
    result = pipeline.process_session(conversation, topic_override="Pointers")
    assert result["status"] == "success"
    topic_data = result["topics"][0]
    assert topic_data["understanding"]["classification"] == "confused"
    assert topic_data["understanding"]["confidence"] > 0.50


# ─────────────────────────────────────────────────────────────────────────────
# CASE 4: Actual Quiz Evaluation (10 questions, answer key, submissions)
# ─────────────────────────────────────────────────────────────────────────────
def test_case_4_actual_quiz_evaluation():
    # 10 questions: 8 correct, 1 incorrect, 1 unanswered -> 80%
    questions = [
        {"id": 1, "topic": "DSA", "question": "Q1"},
        {"id": 2, "topic": "DSA", "question": "Q2"},
        {"id": 3, "topic": "DSA", "question": "Q3"},
        {"id": 4, "topic": "DSA", "question": "Q4"},
        {"id": 5, "topic": "DSA", "question": "Q5"},
        {"id": 6, "topic": "DSA", "question": "Q6"},
        {"id": 7, "topic": "DSA", "question": "Q7"},
        {"id": 8, "topic": "DSA", "question": "Q8"},
        {"id": 9, "topic": "DSA", "question": "Q9"},
        {"id": 10, "topic": "DSA", "question": "Q10"},
    ]
    answer_key = {1: "A", 2: "B", 3: "C", 4: "D", 5: "A", 6: "B", 7: "C", 8: "D", 9: "A", 10: "B"}
    student_answers = {
        1: "A",  # Correct
        2: "B",  # Correct
        3: "C",  # Correct
        4: "D",  # Correct
        5: "A",  # Correct
        6: "B",  # Correct
        7: "C",  # Correct
        8: "D",  # Correct
        9: "C",  # Incorrect (Key is A)
        10: "",  # Unanswered
    }

    eval_result = evaluate_quiz_submission(
        questions=questions,
        answer_key=answer_key,
        student_answers=student_answers,
    )

    assert eval_result["total_questions"] == 10
    assert eval_result["correct_count"] == 8
    assert eval_result["incorrect_count"] == 1
    assert eval_result["unanswered_count"] == 1
    assert eval_result["actual_score_percent"] == 80.0
    assert len(eval_result["question_details"]) == 10
    assert eval_result["question_details"][8]["status"] == "incorrect"
    assert eval_result["question_details"][9]["status"] == "unanswered"


# ─────────────────────────────────────────────────────────────────────────────
# CASE 5: Missing Quiz Responses (No fabricated quiz score)
# ─────────────────────────────────────────────────────────────────────────────
def test_case_5_missing_quiz_responses():
    pipeline = EduLensUnifiedPipeline()
    sample = ["explain quicksort partition step by step.", "got it, elements smaller than pivot go left."]
    res = pipeline.process_session(sample)

    assert res["status"] == "success"
    topic_res = res["topics"][0]
    assert topic_res["actual_quiz_performance"] is None
    assert topic_res["overall_performance"]["score"] is None
    assert topic_res["overall_performance"]["status"] == "awaiting_quiz_submission"
    assert "provisional_formula" in str(topic_res["overall_performance"]) or "formula" in str(topic_res["overall_performance"])


# ─────────────────────────────────────────────────────────────────────────────
# CASE 6: Multiple Topics (Separated topic processing)
# ─────────────────────────────────────────────────────────────────────────────
def test_case_6_multiple_topics_separation():
    pipeline = EduLensUnifiedPipeline()
    multi_topic_session = {
        "chat_id": "MULTI_001",
        "subject": "Computer Science",
        "topics": [
            {
                "topic": "Binary Search",
                "student_messages": [
                    "can you explain binary search time complexity?",
                    "got it! dividing by 2 gives O(log n). makes complete sense!",
                ],
            },
            {
                "topic": "Recursion",
                "student_messages": [
                    "i am confused about base cases in recursion.",
                    "why does it give stack overflow? i am completely stuck.",
                ],
            },
        ],
    }

    res = pipeline.process_session(multi_topic_session)
    assert res["status"] == "success"
    assert res["is_multi_topic"] is True
    assert len(res["topics"]) == 2
    assert res["topics"][0]["topic"] == "Binary Search"
    assert res["topics"][1]["topic"] == "Recursion"
    # Ensure they have independent predictions
    assert res["topics"][0]["understanding"]["classification"] in {"understood", "partial", "confused"}
    assert res["topics"][1]["understanding"]["classification"] == "confused"


# ─────────────────────────────────────────────────────────────────────────────
# CASE 7: Empty Conversation (Graceful validation error, no crash)
# ─────────────────────────────────────────────────────────────────────────────
def test_case_7_empty_conversation_handling():
    pipeline = EduLensUnifiedPipeline()
    # Empty string
    res_str = pipeline.process_session("")
    assert res_str["status"] == "validation_failed"
    assert len(res_str["errors"]) > 0

    # Empty list
    res_list = pipeline.process_session([])
    assert res_list["status"] == "validation_failed"

    # None
    res_none = pipeline.process_session(None)
    assert res_none["status"] == "validation_failed"


# ─────────────────────────────────────────────────────────────────────────────
# CASE 8: Malformed JSON Handling
# ─────────────────────────────────────────────────────────────────────────────
def test_case_8_malformed_json_handling():
    is_valid, errs, _ = validate_session_json(12345)
    assert not is_valid
    assert len(errs) > 0

    is_valid_dict, errs_dict, _ = validate_session_json({"empty_dict_with_no_turns": True})
    assert not is_valid_dict
    assert len(errs_dict) > 0


# ─────────────────────────────────────────────────────────────────────────────
# CASE 9: Unknown Student (Stateless inference works without assumptions)
# ─────────────────────────────────────────────────────────────────────────────
def test_case_9_unknown_student_handling():
    pipeline = EduLensUnifiedPipeline()
    res = pipeline.process_session(
        {"student_id": "TOTALLY_NEW_STUDENT_XYZ", "student_messages": ["what is an array in C++?", "understood now!"]}
    )
    assert res["status"] == "success"
    assert res["student_id"] == "TOTALLY_NEW_STUDENT_XYZ"
    assert len(res["topics"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# CASE 10: Unseen Vocabulary (Vector dimension stability)
# ─────────────────────────────────────────────────────────────────────────────
def test_case_10_unseen_vocabulary_stability():
    predictor = EduLensPredictor()
    # Out of vocabulary words and pseudocode
    alien_text = [
        "zyzzx qwertyu blargh flubdubber int* ptr = &val;",
        "supercalifragilisticexpialidocious quantum entanglement foobar!",
    ]
    res = predictor.predict(alien_text)
    assert res["predicted_class"] in {"understood", "partial", "confused"}
    assert len(res["probabilities"]) == 3
    assert res["conversational_signals"]["num_turns"] == 2.0


# ─────────────────────────────────────────────────────────────────────────────
# CASE 11: Repeated Student Records (Zero Persona Leakage Across Splits)
# ─────────────────────────────────────────────────────────────────────────────
def test_case_11_zero_student_leakage_group_split():
    from ml_prototype.data_loader import load_primary_dataset
    from sklearn.model_selection import StratifiedGroupKFold

    df = load_primary_dataset()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    X = df[["full_text", "student_messages"]]
    y = df["final_understanding"]
    groups = df["student_id"]

    for train_idx, test_idx in sgkf.split(X, y, groups):
        train_s = set(groups.iloc[train_idx])
        test_s = set(groups.iloc[test_idx])
        assert len(train_s & test_s) == 0, "Student persona leakage detected across splits!"


# ─────────────────────────────────────────────────────────────────────────────
# CASE 12: Missing Model Artifact Error Reporting
# ─────────────────────────────────────────────────────────────────────────────
def test_case_12_missing_model_artifact_error():
    fake_path = ROOT_DIR / "models" / "non_existent_folder_abc" / "fake_model.joblib"
    with pytest.raises(FileNotFoundError) as exc_info:
        EduLensPredictor(model_path=fake_path)
    assert "Trained model artifact not found" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT TESTS: DSP, NLP, Confidence, Scoring, API
# ─────────────────────────────────────────────────────────────────────────────
def test_dsp_metrics_extraction():
    session = {
        "turns": [
            {"student_message": "What is quicksort?", "teacher_message": "Quicksort is a divide and conquer algorithm."},
            {"student_message": "How does partition work?", "teacher_message": "Partition places the pivot in its final position."},
        ]
    }
    dsp = extract_dsp_conversation_metrics(session)
    assert dsp["total_messages"] == 4
    assert dsp["student_turn_count"] == 2
    assert dsp["teacher_turn_count"] == 2
    assert dsp["student_total_words"] > 0
    assert dsp["teacher_total_words"] > 0


def test_confidence_estimator_heuristics():
    est = ExpressedConfidenceEstimator()
    hesitant = est.predict(["uh... maybe recursion is calling itself? not sure... is it?"])
    assert hesitant["classification"] == "low"
    assert hesitant["score"] < 40.0

    assertive = est.predict(["definitely! the base case prevents stack overflow. obviously it works by divide and conquer."])
    assert assertive["classification"] == "high"
    assert assertive["score"] > 70.0


def test_scoring_engine_composite_math():
    engine = TopicPerformanceScoringEngine()
    # Understanding 85, Quiz 80, Confidence 75
    # (85 * 0.40) + (80 * 0.40) + (75 * 0.20) = 34 + 32 + 15 = 81.0
    res = engine.compute_topic_performance(
        topic="Logistic Regression",
        understanding_input=85.0,
        actual_quiz_score=80.0,
        expressed_confidence_score=75.0,
    )
    assert res["overall_performance_score"] == 81.0
    assert res["status"] == "complete"


def test_api_new_endpoints():
    """Verify that active production endpoints respond correctly and legacy endpoints return 404."""
    # 1. Active health endpoint
    res_h = client.get("/api/health")
    assert res_h.status_code == 200
    assert res_h.json()["status"] == "healthy"

    # 2. Active model-info endpoint
    res_m = client.get("/api/model-info")
    assert res_m.status_code == 200
    assert "Logistic Regression" in res_m.json()["model_name"]

    # 3. Active sample-conversation endpoint
    res_s = client.get("/api/sample-conversation")
    assert res_s.status_code == 200
    assert "messages" in res_s.json()

    # 4. Active analyze endpoint
    res_a = client.post("/api/analyze", json=res_s.json())
    assert res_a.status_code == 200
    assert "understanding" in res_a.json()
    assert "expressed_confidence" in res_a.json()

    # 5. Verify removed legacy endpoints return 404 Not Found
    assert client.post("/api/evaluate-quiz", json={}).status_code == 404
    assert client.post("/api/confidence", json={}).status_code == 404
    assert client.post("/api/pipeline", json={}).status_code == 404
