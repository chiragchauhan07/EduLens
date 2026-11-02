"""
api/index.py — Vercel Serverless API entrypoint for EduLens.

Exposes RESTful endpoints for real-time model inference, topic performance analytics,
dataset inspection, and model metadata.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml_prototype.predict import EduLensPredictor, CLASS_METADATA
from ml_prototype.data_loader import extract_student_messages, normalize_session

app = FastAPI(
    title="EduLens API",
    description="Student Learning Intelligence & Understanding Analytics API",
    version="1.0.0",
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded predictor and cached data
_predictor: Optional[EduLensPredictor] = None
_cached_topics: Optional[List[Dict[str, Any]]] = None
_cached_eval: Optional[Dict[str, Any]] = None


def get_predictor() -> EduLensPredictor:
    global _predictor
    if _predictor is None:
        model_path = ROOT_DIR / "models" / "trained_model.joblib"
        _predictor = EduLensPredictor(model_path=model_path)
    return _predictor


class PredictRequest(BaseModel):
    conversation: Optional[Union[str, List[str], Dict[str, Any]]] = Field(
        None,
        description="Text transcript, list of student messages, or structured session object with 'turns'/'messages'",
    )
    text: Optional[str] = Field(None, description="Plain text transcript of student messages")


SAMPLE_CONVERSATIONS = [
    {
        "id": "sample-confused",
        "title": "Recursion & Base Cases (Struggling)",
        "subject": "DSA",
        "topic": "Recursion",
        "expected": "confused",
        "messages": [
            "I really don't understand how this recursive function unwinds.",
            "I am completely stuck and confused. What does base case mean? Why does my code crash?",
            "Wait what? Can you explain again step by step? I am totally lost.",
        ],
    },
    {
        "id": "sample-partial",
        "title": "SQL Group By & Having (Partial)",
        "subject": "DBMS",
        "topic": "SQL Queries",
        "expected": "partial",
        "messages": [
            "So WHERE filters rows before aggregation, right?",
            "Can you give another example of when to use HAVING instead of WHERE?",
            "I think I get the basic idea, but how does it behave with null values?",
        ],
    },
    {
        "id": "sample-understood",
        "title": "Binary Search Tree Inorder Traversal (Mastery)",
        "subject": "DSA",
        "topic": "Binary Search Trees",
        "expected": "understood",
        "messages": [
            "Inorder traversal visits left subtree, root node, then right subtree.",
            "For a BST, an inorder traversal always produces keys in strictly ascending sorted order. That makes sense!",
            "Got it! In C++, void inorder(Node* root) { if (!root) return; inorder(root->left); cout << root->val; inorder(root->right); } clear now, thank you!",
        ],
    },
]


@app.get("/api")
@app.get("/api/health")
def health_check():
    """Health check and API metadata endpoint."""
    return {
        "status": "healthy",
        "service": "EduLens API",
        "version": "1.0.0",
        "runtime": "Vercel Serverless Function",
    }


@app.post("/api/predict")
def predict_understanding(req: PredictRequest):
    """
    Evaluates student dialogue and returns predicted comprehension level,
    probabilities, and extracted conversational signals.
    """
    input_data = req.conversation if req.conversation is not None else req.text
    if not input_data:
        raise HTTPException(status_code=400, detail="Either 'conversation' or 'text' must be provided.")

    try:
        predictor = get_predictor()
        result = predictor.predict(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/samples")
def get_samples():
    """Returns pre-built sample learning dialogues for testing."""
    return {"samples": SAMPLE_CONVERSATIONS}


@app.get("/api/topics")
def get_topic_analytics():
    """Returns curriculum-level topic mastery breakdown across CS subjects."""
    global _cached_topics
    if _cached_topics is not None:
        return {"topics": _cached_topics}

    dataset_path = ROOT_DIR / "data" / "raw" / "edulens_chats_dataset.json"
    if not dataset_path.exists():
        return {"topics": [], "warning": "Primary dataset not found on server."}

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        topic_stats: Dict[str, Dict[str, Any]] = {}
        for rec in records:
            subject = rec.get("subject", "General")
            topic = rec.get("topic", "General Topic")
            key = f"{subject}::{topic}"
            label = rec.get("final_understanding", "unknown")
            turns = len(extract_student_messages(rec))

            if key not in topic_stats:
                topic_stats[key] = {
                    "subject": subject,
                    "topic": topic,
                    "total_sessions": 0,
                    "understood": 0,
                    "partial": 0,
                    "confused": 0,
                    "total_turns": 0,
                }

            topic_stats[key]["total_sessions"] += 1
            topic_stats[key]["total_turns"] += turns
            if label in topic_stats[key]:
                topic_stats[key][label] += 1

        summary = []
        for item in topic_stats.values():
            n = item["total_sessions"]
            summary.append({
                "subject": item["subject"],
                "topic": item["topic"],
                "total_sessions": n,
                "understood_count": item["understood"],
                "partial_count": item["partial"],
                "confused_count": item["confused"],
                "mastery_rate": round((item["understood"] / n) * 100, 1) if n else 0,
                "partial_rate": round((item["partial"] / n) * 100, 1) if n else 0,
                "confusion_rate": round((item["confused"] / n) * 100, 1) if n else 0,
                "avg_turns": round(item["total_turns"] / n, 1) if n else 0,
            })

        summary.sort(key=lambda x: (x["subject"], x["topic"]))
        _cached_topics = summary
        return {"topics": _cached_topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate topic summary: {str(e)}")


@app.get("/api/model-info")
def get_model_info():
    """Returns technical model evaluation benchmarks and confusion matrix."""
    global _cached_eval
    if _cached_eval is not None:
        return _cached_eval

    report_path = ROOT_DIR / "reports" / "model_evaluation.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Model evaluation report not found.")

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            _cached_eval = json.load(f)
        return _cached_eval
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model evaluation: {str(e)}")
