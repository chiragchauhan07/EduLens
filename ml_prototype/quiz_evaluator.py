"""
quiz_evaluator.py — Deterministic Actual Quiz Performance Evaluator for EduLens.

Evaluates student-submitted quiz answers against a ground-truth answer key.
This is strictly deterministic and distinct from the conversational quiz score regressor.
"""

from typing import Any, Dict, List, Optional, Union


def evaluate_quiz_submission(
    questions: Optional[List[Dict[str, Any]]] = None,
    answer_key: Optional[Dict[Union[str, int], Any]] = None,
    student_answers: Optional[Dict[Union[str, int], Any]] = None,
    submission_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates student quiz performance deterministically.

    Supported calling conventions:
    1. Direct arguments:
       - questions: List of dicts, each with 'question_id' (or 'id'), 'question_text',
         optional 'correct_answer', and optional 'topic'.
       - answer_key: Dict mapping question_id -> correct_answer.
       - student_answers: Dict mapping question_id -> student_answer.
    2. Unified submission_payload:
       - Dict containing 'questions', 'answer_key', and 'student_answers'
         (or embedded in questions as 'submitted_answer' and 'correct_answer').

    Returns a structured dictionary with:
    - total_questions (int)
    - correct_count (int)
    - incorrect_count (int)
    - unanswered_count (int)
    - actual_score_percent (float, 0.0 to 100.0)
    - topic_breakdown (dict of topic -> {correct, total, score_percent})
    - question_details (list of per-question results)
    """
    if submission_payload is not None:
        if questions is None:
            questions = submission_payload.get("questions")
        if answer_key is None:
            answer_key = submission_payload.get("answer_key", {})
        if student_answers is None:
            student_answers = submission_payload.get("student_answers", {})

    answer_key = answer_key or {}
    student_answers = student_answers or {}

    # If questions list is not provided, reconstruct from answer_key keys
    if not questions:
        if not answer_key and not student_answers:
            return {
                "error": "No quiz questions, answer key, or submitted answers provided.",
                "total_questions": 0,
                "correct_count": 0,
                "incorrect_count": 0,
                "unanswered_count": 0,
                "actual_score_percent": None,
                "topic_breakdown": {},
                "question_details": [],
            }
        all_q_ids = sorted(list(set(list(answer_key.keys()) + list(student_answers.keys()))), key=str)
        questions = [{"question_id": qid} for qid in all_q_ids]

    correct_count = 0
    incorrect_count = 0
    unanswered_count = 0
    question_details = []
    topic_stats: Dict[str, Dict[str, int]] = {}

    for idx, q in enumerate(questions, start=1):
        q_id = q.get("question_id", q.get("id", idx))
        q_text = q.get("question_text", q.get("question", f"Question {idx}"))
        topic = q.get("topic", "General")

        # Determine expected correct answer
        expected = q.get("correct_answer")
        if expected is None:
            expected = answer_key.get(q_id, answer_key.get(str(q_id), answer_key.get(int(q_id) if str(q_id).isdigit() else None)))

        # Determine student submitted answer
        submitted = q.get("submitted_answer", q.get("student_answer"))
        if submitted is None:
            submitted = student_answers.get(q_id, student_answers.get(str(q_id), student_answers.get(int(q_id) if str(q_id).isdigit() else None)))

        if topic not in topic_stats:
            topic_stats[topic] = {"total": 0, "correct": 0, "incorrect": 0, "unanswered": 0}
        topic_stats[topic]["total"] += 1

        # Check unanswered
        if submitted is None or (isinstance(submitted, str) and submitted.strip() == ""):
            status = "unanswered"
            is_correct = False
            unanswered_count += 1
            topic_stats[topic]["unanswered"] += 1
        else:
            # Normalize comparisons (strings stripped and case-insensitive)
            norm_sub = str(submitted).strip().lower()
            norm_exp = str(expected).strip().lower() if expected is not None else None

            if norm_exp is not None and norm_sub == norm_exp:
                status = "correct"
                is_correct = True
                correct_count += 1
                topic_stats[topic]["correct"] += 1
            else:
                status = "incorrect"
                is_correct = False
                incorrect_count += 1
                topic_stats[topic]["incorrect"] += 1

        question_details.append({
            "question_id": q_id,
            "question_text": q_text,
            "topic": topic,
            "submitted_answer": submitted,
            "correct_answer": expected,
            "status": status,
            "is_correct": is_correct,
        })

    total_questions = len(question_details)
    actual_score_percent = (
        round((correct_count / total_questions) * 100.0, 2)
        if total_questions > 0
        else 0.0
    )

    topic_breakdown = {}
    for t_name, stats in topic_stats.items():
        t_total = stats["total"]
        topic_breakdown[t_name] = {
            "total_questions": t_total,
            "correct_count": stats["correct"],
            "incorrect_count": stats["incorrect"],
            "unanswered_count": stats["unanswered"],
            "score_percent": round((stats["correct"] / t_total) * 100.0, 2) if t_total > 0 else 0.0,
        }

    return {
        "total_questions": total_questions,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "unanswered_count": unanswered_count,
        "actual_score_percent": actual_score_percent,
        "topic_breakdown": topic_breakdown,
        "question_details": question_details,
        "question_breakdown": question_details,
    }

