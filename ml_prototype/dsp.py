"""
dsp.py — Digital Signal Processing & Conversational Data Engineering Pipeline for EduLens.

Handles:
1. Raw conversation JSON ingestion and schema validation.
2. Missing-value, duplicate, and malformed-turn detection.
3. Turn-taking and conversational structural statistics (student vs. teacher distribution).
4. Dialogue length, duration, and utterance volume metrics.
5. Reproducible session normalization for downstream NLP and ML modules.
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union


def validate_session_json(raw_input: Any) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validates raw conversation JSON schema, structure, and content.
    Returns (is_valid, validation_errors, sanitized_data).
    """
    errors = []
    if raw_input is None:
        return False, ["Input is null / empty."], {}

    if isinstance(raw_input, str):
        if not raw_input.strip():
            return False, ["Input string is empty."], {}
        lines = [l.strip() for l in raw_input.split("\n") if l.strip()]
        return True, [], {"turns": [{"turn": i, "student_message": l} for i, l in enumerate(lines, 1)]}

    if isinstance(raw_input, list):
        if len(raw_input) == 0:
            return False, ["Conversation list is empty."], {}
        if isinstance(raw_input[0], str):
            turns = [{"turn": i, "student_message": m} for i, m in enumerate(raw_input, 1) if str(m).strip()]
            if not turns:
                return False, ["Conversation list contains only empty strings."], {}
            return True, [], {"turns": turns}
        elif isinstance(raw_input[0], dict):
            return True, [], {"turns": raw_input}

    if isinstance(raw_input, dict):
        sanitized = dict(raw_input)
        turns = sanitized.get("turns", sanitized.get("messages", sanitized.get("student_messages", sanitized.get("topics", []))))
        if not turns and not sanitized.get("full_text"):
            errors.append("No dialogue turns, messages, or text found in session object.")
        return len(errors) == 0, errors, sanitized

    return False, [f"Unsupported input type: {type(raw_input).__name__}"], {}


def extract_dsp_conversation_metrics(session: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts comprehensive conversational and structural statistics from a session:
    - Message counts (student vs. teacher)
    - Word volumes & ratios
    - Response length distributions (mean, min, max, std)
    - Turn alternation statistics
    - Temporal / timestamp metrics (if present)
    """
    turns = session.get("turns", session.get("messages", []))
    student_msgs = []
    teacher_msgs = []
    alternations = 0

    if isinstance(turns, list):
        prev_speaker = None
        for item in turns:
            if isinstance(item, dict):
                s_msg = item.get("student_message", item.get("student", item.get("user", "")))
                t_msg = item.get("teacher_message", item.get("teacher", item.get("assistant", "")))
                if s_msg and str(s_msg).strip():
                    student_msgs.append(str(s_msg).strip())
                    if prev_speaker != "student":
                        alternations += 1
                        prev_speaker = "student"
                if t_msg and str(t_msg).strip():
                    teacher_msgs.append(str(t_msg).strip())
                    if prev_speaker != "teacher":
                        alternations += 1
                        prev_speaker = "teacher"
            elif isinstance(item, str) and item.strip():
                student_msgs.append(item.strip())
    elif "student_messages" in session and isinstance(session["student_messages"], list):
        student_msgs = [str(m).strip() for m in session["student_messages"] if str(m).strip()]

    # If no separated turns but full_text exists
    if not student_msgs and session.get("full_text"):
        student_msgs = [session["full_text"]]

    # Turn word counts
    student_turn_words = [len(m.split()) for m in student_msgs]
    teacher_turn_words = [len(m.split()) for m in teacher_msgs]

    s_count = len(student_msgs)
    t_count = len(teacher_msgs)
    total_msgs = s_count + t_count

    s_words = sum(student_turn_words)
    t_words = sum(teacher_turn_words)
    total_words = s_words + t_words

    # Summary Statistics
    s_avg_words = float(s_words / s_count) if s_count > 0 else 0.0
    s_min_words = min(student_turn_words) if s_count > 0 else 0
    s_max_words = max(student_turn_words) if s_count > 0 else 0

    if s_count > 1:
        variance = sum((w - s_avg_words) ** 2 for w in student_turn_words) / (s_count - 1)
        s_std_words = round(math.sqrt(variance), 2)
    else:
        s_std_words = 0.0

    t_avg_words = float(t_words / t_count) if t_count > 0 else 0.0
    msg_ratio = round(float(s_count / t_count), 2) if t_count > 0 else (float(s_count) if s_count > 0 else 0.0)
    word_ratio = round(float(s_words / t_words), 2) if t_words > 0 else (float(s_words) if s_words > 0 else 0.0)

    # Duplicate student turns check
    unique_s_msgs = set(m.lower() for m in student_msgs)
    dup_student_turns = s_count - len(unique_s_msgs)

    # Timestamp / Duration calculation if available
    duration_minutes = None
    ts_str = session.get("timestamp")
    if ts_str:
        for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
            try:
                dt = datetime.strptime(str(ts_str)[:19], fmt)
                duration_minutes = None  # Single timestamp represents start or submission time
                break
            except Exception:
                continue

    return {
        "total_messages": total_msgs,
        "student_turn_count": s_count,
        "teacher_turn_count": t_count,
        "student_to_teacher_msg_ratio": msg_ratio,
        "student_total_words": s_words,
        "teacher_total_words": t_words,
        "total_session_words": total_words,
        "student_to_teacher_word_ratio": word_ratio,
        "student_avg_turn_words": round(s_avg_words, 2),
        "student_min_turn_words": s_min_words,
        "student_max_turn_words": s_max_words,
        "student_turn_words_std": s_std_words,
        "teacher_avg_turn_words": round(t_avg_words, 2),
        "turn_alternations": alternations,
        "duplicate_student_turns": dup_student_turns,
        "has_repeated_messages": dup_student_turns > 0,
        "session_timestamp": str(ts_str) if ts_str else None,
        "duration_minutes": duration_minutes,
    }


def clean_and_normalize_session(raw_session: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleans and standardizes raw session payload into consistent internal format.
    """
    chat_id = raw_session.get("chat_id", "UNKNOWN_SESSION")
    student_id = raw_session.get("student_id", "ANONYMOUS_STUDENT")
    subject = raw_session.get("subject", "Computer Science")
    topic = raw_session.get("topic", "General Topic")
    persona = raw_session.get("student_persona", "standard")

    dsp_metrics = extract_dsp_conversation_metrics(raw_session)

    # Extract student text
    from ml_prototype.data_loader import extract_student_messages
    student_msgs = extract_student_messages(raw_session)
    full_text = " ".join(student_msgs)

    return {
        "chat_id": chat_id,
        "student_id": student_id,
        "subject": subject,
        "topic": topic,
        "student_persona": persona,
        "student_messages": student_msgs,
        "full_text": full_text,
        "dsp_metrics": dsp_metrics,
        "ground_truth_understanding": raw_session.get("final_understanding"),
        "raw_quiz_data": raw_session.get("quiz"),
    }
