"""
data_loader.py — Dataset loading and normalization for EduLens ML Prototype.

This module loads raw conversation datasets and converts them into a clean,
standardized representation suitable for feature extraction and model training.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


def extract_student_messages(record: Dict[str, Any]) -> List[str]:
    """
    Extracts student utterances from a conversation record regardless of schema variation.
    
    Handles:
    - `turns`: List of dicts with 'student_message' (e.g., edulens_chats_dataset.json)
    - `messages`: List of dicts with 'student_message' (e.g., set1, set2, set3)
    - `conversation`: String or list of turns
    """
    student_messages: List[str] = []
    
    # Format 1: 'turns' array with 'student_message'
    if "turns" in record and isinstance(record["turns"], list):
        for turn in record["turns"]:
            if isinstance(turn, dict) and "student_message" in turn:
                msg = str(turn["student_message"]).strip()
                if msg:
                    student_messages.append(msg)
                    
    # Format 2: 'messages' array with 'student_message'
    elif "messages" in record and isinstance(record["messages"], list):
        for turn in record["messages"]:
            if isinstance(turn, dict) and "student_message" in turn:
                msg = str(turn["student_message"]).strip()
                if msg:
                    student_messages.append(msg)
                    
    # Format 3: Raw string or flat list
    elif "student_messages" in record and isinstance(record["student_messages"], list):
        student_messages = [str(m).strip() for m in record["student_messages"] if str(m).strip()]
        
    return student_messages


def normalize_session(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes a single conversation record into standard dictionary structure.
    """
    student_msgs = extract_student_messages(record)
    full_text = " ".join(student_msgs)
    
    return {
        "chat_id": record.get("chat_id", "UNKNOWN"),
        "student_id": record.get("student_id", "UNKNOWN"),
        "subject": record.get("subject", "General"),
        "topic": record.get("topic", "General Topic"),
        "student_persona": record.get("student_persona", "general"),
        "student_messages": student_msgs,
        "full_text": full_text,
        "num_turns": len(student_msgs),
        "final_understanding": record.get("final_understanding", None),
    }


def load_primary_dataset(filepath: Union[str, Path] = "data/raw/edulens_chats_dataset.json") -> pd.DataFrame:
    """
    Loads and normalizes the primary EduLens dataset with supervised learning labels.
    
    Returns:
        pd.DataFrame containing standardized conversation fields and target labels.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Primary dataset not found at {path.resolve()}")
        
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    normalized = [normalize_session(rec) for rec in raw_data]
    df = pd.DataFrame(normalized)
    
    # Filter out empty conversations or missing targets
    valid_df = df[df["num_turns"] > 0].copy()
    valid_df = valid_df[valid_df["final_understanding"].notna()].copy()
    
    return valid_df


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("Testing data_loader.py...")
    df = load_primary_dataset()
    print(f"Loaded {len(df)} sessions.")
    print("Class distribution:")
    print(df["final_understanding"].value_counts())
    print("\nSample record:")
    sample = df.iloc[0]
    print(f"Chat ID: {sample['chat_id']}, Subject: {sample['subject']}, Topic: {sample['topic']}")
    print(f"Turns: {sample['num_turns']}, Final Understanding: {sample['final_understanding']}")
    clean_preview = sample['full_text'][:120].encode('ascii', errors='replace').decode('ascii')
    print(f"Text preview: {clean_preview}...")
