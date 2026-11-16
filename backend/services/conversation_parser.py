"""
conversation_parser.py — Robust JSON conversation parser and validator for EduLens.

Handles:
- Multiple JSON schemas (standard messages list, turns array, separate student_turns/tutor_turns, raw message lists)
- Varied role naming ('student', 'learner', 'user' vs 'assistant', 'ai', 'tutor', 'teacher')
- Varied text fields ('content', 'text', 'message', 'student_message', 'utterance')
- Safe validation for empty payloads, missing student turns, oversized files, invalid JSON.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB limit

STUDENT_ROLES = {"student", "learner", "user", "pupil", "human"}
ASSISTANT_ROLES = {"assistant", "ai", "tutor", "teacher", "bot", "system"}


@dataclass
class ParsedConversation:
    topic: str
    student_messages: List[str]
    assistant_messages: List[str]
    student_turn_count: int
    assistant_turn_count: int
    total_turns: int
    raw_topic_found: bool


def _clean_str(val: Any) -> str:
    if val is None:
        return ""
    return str(val).strip()


def parse_conversation_json(
    raw_data: Union[str, bytes, Dict[str, Any], List[Any]],
    override_topic: Optional[str] = None,
) -> ParsedConversation:
    """
    Parses and validates an uploaded educational conversation payload.

    Args:
        raw_data: JSON string, raw bytes, or already-decoded dictionary/list.
        override_topic: Optional topic string entered manually by user.

    Returns:
        ParsedConversation object with extracted student and assistant messages.

    Raises:
        ValueError: On schema violations, missing student messages, or invalid format.
    """
    # 1. Parse JSON if string or bytes
    if isinstance(raw_data, bytes):
        if len(raw_data) > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"Uploaded file exceeds the maximum allowed size of 5MB ({len(raw_data)} bytes).")
        if not raw_data.strip():
            raise ValueError("Uploaded file is empty.")
        try:
            payload = json.loads(raw_data.decode("utf-8"))
        except UnicodeDecodeError:
            try:
                payload = json.loads(raw_data.decode("latin-1"))
            except Exception as e:
                raise ValueError(f"File encoding error: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON syntax: {str(e)}")
    elif isinstance(raw_data, str):
        if not raw_data.strip():
            raise ValueError("Uploaded conversation text is empty.")
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON syntax: {str(e)}")
    elif isinstance(raw_data, (dict, list)):
        payload = raw_data
    else:
        raise ValueError(f"Unsupported payload type: {type(raw_data).__name__}")

    student_messages: List[str] = []
    assistant_messages: List[str] = []
    extracted_topic = ""
    raw_topic_found = False

    # 2. Extract Topic if payload is a dict
    if isinstance(payload, dict):
        for topic_key in ["topic", "subject", "concept", "concept_domain", "title", "lesson"]:
            if topic_key in payload and _clean_str(payload[topic_key]):
                extracted_topic = _clean_str(payload[topic_key])
                raw_topic_found = True
                break

    # Determine final topic
    final_topic = (
        _clean_str(override_topic)
        if override_topic and _clean_str(override_topic)
        else (extracted_topic if extracted_topic else "General / Not Specified")
    )

    # 3. Schema Variations

    # Case A: Top-level list of message dictionaries: [{"role": "student", "content": "..."}]
    if isinstance(payload, list):
        _extract_from_message_list(payload, student_messages, assistant_messages)

    elif isinstance(payload, dict):
        # Case B: 'messages' or 'dialogue' or 'conversation' list
        for list_key in ["messages", "dialogue", "conversation", "chat", "history", "dialogue_history"]:
            if list_key in payload and isinstance(payload[list_key], list):
                _extract_from_message_list(payload[list_key], student_messages, assistant_messages)
                break

        # Case C: 'turns' list (standard EduLens dataset format)
        if not student_messages and "turns" in payload and isinstance(payload["turns"], list):
            for turn in payload["turns"]:
                if isinstance(turn, dict):
                    # Format: turn has 'student_message' and/or 'assistant_message'
                    if "student_message" in turn and _clean_str(turn["student_message"]):
                        student_messages.append(_clean_str(turn["student_message"]))
                    if "assistant_message" in turn and _clean_str(turn["assistant_message"]):
                        assistant_messages.append(_clean_str(turn["assistant_message"]))
                    elif "tutor_message" in turn and _clean_str(turn["tutor_message"]):
                        assistant_messages.append(_clean_str(turn["tutor_message"]))
                    # Format: turn has role/speaker and text/content
                    elif "role" in turn or "speaker" in turn:
                        _process_single_message_dict(turn, student_messages, assistant_messages)

        # Case D: Separate 'student_turns' and 'tutor_turns' / 'assistant_turns'
        if not student_messages and ("student_turns" in payload or "tutor_turns" in payload):
            if "student_turns" in payload and isinstance(payload["student_turns"], list):
                for item in payload["student_turns"]:
                    text = _extract_text_field(item)
                    if text:
                        student_messages.append(text)
            if "tutor_turns" in payload and isinstance(payload["tutor_turns"], list):
                for item in payload["tutor_turns"]:
                    text = _extract_text_field(item)
                    if text:
                        assistant_messages.append(text)
            elif "assistant_turns" in payload and isinstance(payload["assistant_turns"], list):
                for item in payload["assistant_turns"]:
                    text = _extract_text_field(item)
                    if text:
                        assistant_messages.append(text)

        # Case E: Direct 'student_messages' list
        if not student_messages and "student_messages" in payload and isinstance(payload["student_messages"], list):
            for m in payload["student_messages"]:
                clean = _clean_str(m)
                if clean:
                    student_messages.append(clean)

        # Case F: Flat transcript text
        if not student_messages and "text" in payload and isinstance(payload["text"], str):
            lines = [l.strip() for l in payload["text"].split("\n") if l.strip()]
            student_messages.extend(lines)

    # 4. Rigorous Validation
    if not student_messages:
        if assistant_messages:
            raise ValueError(
                "Conversation contains only assistant/tutor messages and zero student turns. "
                "Student utterances are required to evaluate comprehension."
            )
        raise ValueError(
            "No student messages could be found. Please ensure your JSON includes a 'messages' list, "
            "'turns' list, or 'student_turns' with 'student' role or 'student_message' keys."
        )

    return ParsedConversation(
        topic=final_topic,
        student_messages=student_messages,
        assistant_messages=assistant_messages,
        student_turn_count=len(student_messages),
        assistant_turn_count=len(assistant_messages),
        total_turns=len(student_messages) + len(assistant_messages),
        raw_topic_found=raw_topic_found,
    )


def _extract_text_field(item: Any) -> str:
    """Extracts text content from a string or dictionary."""
    if isinstance(item, str):
        return _clean_str(item)
    if isinstance(item, dict):
        for field in ["text", "content", "message", "utterance", "student_message", "tutor_message", "body"]:
            if field in item and _clean_str(item[field]):
                return _clean_str(item[field])
    return ""


def _process_single_message_dict(
    msg: Dict[str, Any],
    student_msgs: List[str],
    assistant_msgs: List[str],
) -> None:
    role = (
        msg.get("role")
        or msg.get("speaker")
        or msg.get("sender")
        or msg.get("author")
        or msg.get("user_type")
        or ""
    )
    role_str = str(role).strip().lower()

    text = _extract_text_field(msg)
    if not text:
        return

    if role_str in STUDENT_ROLES:
        student_msgs.append(text)
    elif role_str in ASSISTANT_ROLES:
        assistant_msgs.append(text)
    elif "student" in role_str:
        student_msgs.append(text)
    elif any(r in role_str for r in ["tutor", "teacher", "ai", "assistant", "bot"]):
        assistant_msgs.append(text)
    else:
        # Default fallback: if first utterance or unmarked, classify as student
        if not student_msgs:
            student_msgs.append(text)
        else:
            assistant_msgs.append(text)


def _extract_from_message_list(
    messages: List[Any],
    student_msgs: List[str],
    assistant_msgs: List[str],
) -> None:
    for item in messages:
        if isinstance(item, dict):
            _process_single_message_dict(item, student_msgs, assistant_msgs)
        elif isinstance(item, str) and _clean_str(item):
            student_msgs.append(_clean_str(item))

