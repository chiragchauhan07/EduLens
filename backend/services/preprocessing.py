"""
preprocessing.py — NLP Preprocessing for EduLens.

Standardizes conversation text, cleans tokens, and formats data frames for the
trained Scikit-Learn inference pipeline.
"""

import re
from typing import Dict, List, Tuple
import pandas as pd


def clean_text(text: str) -> str:
    """
    Cleans utterance text by standardizing whitespace and removing control chars
    while preserving pedagogical punctuation (question marks, quotes).
    """
    if not text:
        return ""
    # Standardize unicode quotes / apostrophes
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    # Collapse multiple spaces and trim
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_inference_frame(student_messages: List[str]) -> Tuple[pd.DataFrame, str]:
    """
    Converts a list of student utterance strings into the exact two-column DataFrame
    format expected by the fitted ColumnTransformer in EduLens models:
    - 'full_text': space-joined cleaned string of all student utterances
    - 'student_messages': list of individual student utterance strings
    """
    cleaned_messages = [clean_text(m) for m in student_messages if clean_text(m)]
    if not cleaned_messages:
        cleaned_messages = [clean_text(m) for m in student_messages]

    full_text = " ".join(cleaned_messages)

    df = pd.DataFrame([{
        "full_text": full_text,
        "student_messages": cleaned_messages,
    }])

    return df, full_text

