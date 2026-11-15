"""
scripts/prepare_data.py — Data Science pipeline for cleaning, auditing, and preparing EduLens datasets.

Workflow:
1. Ingests raw primary dataset ('data/raw/edulens_chats_dataset.json') and supplementary sets.
2. Performs data inspection, missing value filtering, and schema validation.
3. Computes descriptive statistics (class balance, dialogue length, turn distributions, vocabulary stats).
4. Exports verified, cleaned datasets to 'data/processed/':
   - 'edulens_clean_primary.json': 430 normalized sessions for understanding classification.
   - 'edulens_clean_quiz.json': Combined sessions with valid quiz scores for regression.
5. Saves an EDA summary report to 'reports/eda_summary.json'.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml_prototype.data_loader import extract_student_messages, normalize_session


def prepare_datasets(
    raw_dir: Path = ROOT_DIR / "data" / "raw",
    processed_dir: Path = ROOT_DIR / "data" / "processed",
    reports_dir: Path = ROOT_DIR / "reports",
) -> Dict[str, Any]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EDULENS DATA SCIENCE PIPELINE — DATA PREPARATION & EDA")
    print("=" * 70)

    # 1. Load and Clean Primary Dataset
    primary_path = raw_dir / "edulens_chats_dataset.json"
    if not primary_path.exists():
        raise FileNotFoundError(f"Primary dataset not found at {primary_path}")

    with open(primary_path, "r", encoding="utf-8") as f:
        primary_raw = json.load(f)

    print(f"\n[1/4] Ingested {len(primary_raw)} raw primary sessions from {primary_path.name}")

    primary_cleaned: List[Dict[str, Any]] = []
    class_counts = Counter()
    subject_counts = Counter()
    topic_counts = Counter()
    turn_lengths: List[int] = []
    word_counts: List[int] = []

    for r in primary_raw:
        norm = normalize_session(r)
        if norm["num_turns"] == 0 or not norm["final_understanding"]:
            continue  # Drop empty or unlabeled records

        student_msgs = norm["student_messages"]
        num_turns = norm["num_turns"]
        total_w = sum(len(m.split()) for m in student_msgs)

        record = {
            "chat_id": norm["chat_id"],
            "student_id": norm["student_id"],
            "subject": norm["subject"],
            "topic": norm["topic"],
            "student_persona": norm["student_persona"],
            "num_turns": num_turns,
            "total_words": total_w,
            "student_messages": student_msgs,
            "full_text": norm["full_text"],
            "final_understanding": norm["final_understanding"],
            "quiz_score_percent": r.get("quiz", {}).get("score_percent") if isinstance(r.get("quiz"), dict) else None,
        }
        primary_cleaned.append(record)

        class_counts[norm["final_understanding"]] += 1
        subject_counts[norm["subject"]] += 1
        topic_counts[norm["topic"]] += 1
        turn_lengths.append(num_turns)
        word_counts.append(total_w)

    clean_primary_path = processed_dir / "edulens_clean_primary.json"
    with open(clean_primary_path, "w", encoding="utf-8") as f:
        json.dump(primary_cleaned, f, indent=2, ensure_ascii=False)
    print(f"      Saved {len(primary_cleaned)} cleaned primary records to: {clean_primary_path.name}")

    # 2. Ingest and Clean Quiz Supplementary Sets
    print("\n[2/4] Processing supplementary sets for quiz score regression...")
    quiz_records: List[Dict[str, Any]] = []

    # First add primary records that have quiz scores
    for r in primary_cleaned:
        if r["quiz_score_percent"] is not None:
            quiz_records.append({
                "source": "primary",
                "chat_id": r["chat_id"],
                "student_id": r["student_id"],
                "subject": r["subject"],
                "topic": r["topic"],
                "student_messages": r["student_messages"],
                "full_text": r["full_text"],
                "num_turns": r["num_turns"],
                "total_words": r["total_words"],
                "quiz_score_percent": float(r["quiz_score_percent"]),
            })

    # Next add supplementary sets 1, 2, 3
    for set_file in ["edulens_chats_set1 (1).json", "edulens_chats_set2 (1).json", "edulens_chats_set3 (1).json"]:
        set_path = raw_dir / set_file
        if not set_path.exists():
            continue
        with open(set_path, "r", encoding="utf-8") as f:
            supp_data = json.load(f)

        for rec in supp_data:
            msgs = extract_student_messages(rec)
            if not msgs:
                continue
            quiz = rec.get("quiz", {})
            score = quiz.get("score_percent") if isinstance(quiz, dict) else None
            if score is None:
                continue

            full_t = " ".join(msgs)
            quiz_records.append({
                "source": set_file,
                "chat_id": rec.get("chat_id", "UNKNOWN"),
                "student_id": rec.get("student_id", "UNKNOWN"),
                "subject": rec.get("subject", "CS"),
                "topic": rec.get("topic", "General"),
                "student_messages": msgs,
                "full_text": full_t,
                "num_turns": len(msgs),
                "total_words": len(full_t.split()),
                "quiz_score_percent": float(score),
            })

    clean_quiz_path = processed_dir / "edulens_clean_quiz.json"
    with open(clean_quiz_path, "w", encoding="utf-8") as f:
        json.dump(quiz_records, f, indent=2, ensure_ascii=False)
    print(f"      Saved {len(quiz_records)} records with quiz scores to: {clean_quiz_path.name}")

    # 3. Compute Comprehensive EDA Statistics
    print("\n[3/4] Computing descriptive statistics and distribution metrics...")
    turn_arr = np.array(turn_lengths)
    word_arr = np.array(word_counts)
    quiz_scores = [r["quiz_score_percent"] for r in quiz_records]
    quiz_arr = np.array(quiz_scores)

    eda_summary = {
        "dataset_name": "EduLens Cleaned Primary Dataset",
        "total_records": len(primary_cleaned),
        "target_class_distribution": dict(class_counts),
        "target_class_proportions": {
            cls: round(cnt / len(primary_cleaned), 4) for cls, cnt in class_counts.items()
        },
        "subject_distribution": dict(subject_counts),
        "num_unique_topics": len(topic_counts),
        "num_unique_students": len(set(r["student_id"] for r in primary_cleaned)),
        "conversational_length_metrics": {
            "turns_mean": round(float(turn_arr.mean()), 2),
            "turns_std": round(float(turn_arr.std()), 2),
            "turns_min": int(turn_arr.min()),
            "turns_max": int(turn_arr.max()),
            "words_mean": round(float(word_arr.mean()), 2),
            "words_std": round(float(word_arr.std()), 2),
            "words_min": int(word_arr.min()),
            "words_max": int(word_arr.max()),
        },
        "quiz_score_metrics": {
            "total_records": len(quiz_records),
            "score_mean": round(float(quiz_arr.mean()), 2),
            "score_std": round(float(quiz_arr.std()), 2),
            "score_min": float(quiz_arr.min()),
            "score_max": float(quiz_arr.max()),
            "score_median": float(np.median(quiz_arr)),
        },
    }

    report_path = reports_dir / "eda_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)
    print(f"      Exported EDA report to: {report_path.name}")

    # 4. Summary Display
    print("\n[4/4] Data Science Pipeline Execution Summary:")
    print(f"      - Total Valid Classification Sessions : {eda_summary['total_records']}")
    print(f"      - Class Distribution                   : {eda_summary['target_class_distribution']}")
    print(f"      - Average Turns per Session            : {eda_summary['conversational_length_metrics']['turns_mean']} turns")
    print(f"      - Average Words per Session            : {eda_summary['conversational_length_metrics']['words_mean']} words")
    print(f"      - Total Records for Quiz Regression    : {eda_summary['quiz_score_metrics']['total_records']}")
    print("=" * 70)

    return eda_summary


if __name__ == "__main__":
    prepare_datasets()

