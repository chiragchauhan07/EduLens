"""
nlp.py — Natural Language Processing & Conversational Vectorization Pipeline for EduLens.

Handles:
1. Student vs. teacher utterance isolation.
2. Text normalization, whitespace cleaning, and special character sanitization.
3. Tokenization and stop-word filtering.
4. Lexical TF-IDF N-gram feature extraction (100 features).
5. Pedagogical conversational signal extraction (8 dense metrics).
6. 108-dimensional numerical feature vector generation for machine learning inference.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml_prototype.features import (
    extract_conversation_stats,
    build_feature_pipeline,
)

# Regular expressions for text normalization
RE_WHITESPACE = re.compile(r"\s+")
RE_SPECIAL_PUNCT = re.compile(r"['`’]")


def normalize_conversational_text(text: str) -> str:
    """
    Cleans and standardizes conversational text while preserving pedagogical symbols.
    - Preserves technical operators (*, &, ->, ==, !=, {}, []) for code analysis.
    - Normalizes curly quotes and unicode punctuation.
    - Trims and collapses redundant whitespace.
    """
    if not text:
        return ""

    # Normalize unicode characters (e.g. smart quotes)
    normalized = unicodedata.normalize("NFKD", text)
    # Replace stylized apostrophes with standard single quote
    normalized = RE_SPECIAL_PUNCT.sub("'", normalized)
    # Collapse whitespace
    cleaned = RE_WHITESPACE.sub(" ", normalized).strip()
    return cleaned.lower()


def parse_and_clean_conversation(
    conversation_input: Union[Dict[str, Any], List[str], str],
) -> Tuple[List[str], List[str], str]:
    """
    Isolates student messages, teacher messages, and builds normalized student text.
    Returns (student_messages, teacher_messages, normalized_student_full_text).
    """
    student_msgs: List[str] = []
    teacher_msgs: List[str] = []

    if isinstance(conversation_input, dict):
        turns = conversation_input.get("turns", conversation_input.get("messages", []))
        if isinstance(turns, list):
            for t in turns:
                if isinstance(t, dict):
                    s = t.get("student_message", t.get("student", t.get("user", "")))
                    tch = t.get("teacher_message", t.get("teacher", t.get("assistant", "")))
                    if s and str(s).strip():
                        student_msgs.append(normalize_conversational_text(str(s)))
                    if tch and str(tch).strip():
                        teacher_msgs.append(normalize_conversational_text(str(tch)))
                elif isinstance(t, str) and t.strip():
                    student_msgs.append(normalize_conversational_text(t))
        elif "student_messages" in conversation_input and isinstance(conversation_input["student_messages"], list):
            student_msgs = [
                normalize_conversational_text(str(m))
                for m in conversation_input["student_messages"]
                if str(m).strip()
            ]

        # Fallback to full_text if present
        if not student_msgs and conversation_input.get("full_text"):
            student_msgs = [normalize_conversational_text(conversation_input["full_text"])]

    elif isinstance(conversation_input, list):
        for item in conversation_input:
            if isinstance(item, str) and item.strip():
                student_msgs.append(normalize_conversational_text(item))
            elif isinstance(item, dict):
                s = item.get("student_message", item.get("student", item.get("user", "")))
                if s and str(s).strip():
                    student_msgs.append(normalize_conversational_text(str(s)))

    elif isinstance(conversation_input, str):
        lines = [normalize_conversational_text(l) for l in conversation_input.split("\n") if l.strip()]
        student_msgs = lines

    full_text = " ".join(student_msgs)
    return student_msgs, teacher_msgs, full_text


def extract_nlp_features_vector(
    pipeline: Pipeline,
    student_messages: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts the fixed 108-dimensional feature vector using the fitted pipeline's preprocessor.
    Guarantees exact dimension and transformer alignment with training.
    """
    if not student_messages:
        # 108-dim zero vector fallback
        return np.zeros((1, 108), dtype=np.float32), []

    full_text = " ".join(student_messages)
    input_df = pd.DataFrame([{
        "full_text": full_text,
        "student_messages": student_messages,
    }])

    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        raise ValueError("Pipeline does not contain a 'preprocessor' transformer stage.")

    X_trans = preprocessor.transform(input_df)
    if hasattr(X_trans, "toarray"):
        vec = X_trans.toarray()
    else:
        vec = np.asarray(X_trans)

    feature_names = []
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            pass

    return vec, feature_names

