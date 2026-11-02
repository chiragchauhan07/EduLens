"""
api/index.py — Vercel Serverless API entrypoint for EduLens.

Exposes RESTful endpoints for real-time model inference, topic performance analytics,
dataset inspection, and model metadata. Supports both root and /api prefixed routes
to ensure full compatibility with Vercel serverless ASGI routing.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import APIRouter, FastAPI, HTTPException, Request
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
_cached_raw_dataset: Optional[List[Dict[str, Any]]] = None


def get_predictor() -> EduLensPredictor:
    global _predictor
    if _predictor is None:
        model_path = ROOT_DIR / "models" / "trained_model.joblib"
        _predictor = EduLensPredictor(model_path=model_path)
    return _predictor


def get_raw_dataset() -> List[Dict[str, Any]]:
    global _cached_raw_dataset
    if _cached_raw_dataset is None:
        dataset_path = ROOT_DIR / "data" / "raw" / "edulens_chats_dataset.json"
        if dataset_path.exists():
            with open(dataset_path, "r", encoding="utf-8") as f:
                _cached_raw_dataset = json.load(f)
        else:
            _cached_raw_dataset = []
    return _cached_raw_dataset


class PredictRequest(BaseModel):
    conversation: Optional[Union[str, List[str], Dict[str, Any]]] = Field(
        None,
        description="Text transcript, list of student messages, or structured session object with 'turns'/'messages'",
    )
    text: Optional[str] = Field(None, description="Plain text transcript of student messages")


SAMPLE_CONVERSATIONS = [
    {
        "id": "sample-0",
        "title": "Sample 1 (C++ Pointers — Struggling/Confused)",
        "subject": "C++",
        "topic": "Pointers",
        "expected": "confused",
        "dataset_index": 0,
        "messages": [
            "explain pointers and references in C++ the way it is asked in university exams please ?",
            "can you show a quick dry run for this snippet ? i get confused tracing values .",
            "wait , why did the value change here ? what is happening with the address ? i am lost .",
            "is there a simpler way to remember this rule for midterms ? please repeat .",
            "thank you , will revise this again before the lab test tomorrow !",
        ],
    },
    {
        "id": "sample-8",
        "title": "Sample 2 (DAA Binary Search — Mastered/Understood)",
        "subject": "DAA",
        "topic": "Divide and Conquer: Binary Search",
        "expected": "understood",
        "dataset_index": 8,
        "messages": [
            "can we go through binary search algorithm in DAA ?",
            "what is the recurrence relation and best / worst case time complexity ?",
            "so T(n) = T(n/2) + O(1) gives O(log n) using Master Theorem Case 2, right?",
            "got it! When array is already sorted, mid = low + (high - low) / 2 prevents integer overflow.",
            "understood completely, that makes sense now! thank you!",
        ],
    },
    {
        "id": "sample-25",
        "title": "Sample 3 (OS Virtual Memory — Emerging/Partial)",
        "subject": "OS",
        "topic": "Virtual Memory and Paging",
        "expected": "partial",
        "dataset_index": 25,
        "messages": [
            "how does page table translation work in virtual memory ?",
            "what causes a page fault and how does the OS handle it ?",
            "so the TLB is like a hardware cache for page table entries, right?",
            "can you give another example of inverted page tables? I understand regular paging but inverted is still unclear.",
            "okay that clarifies some doubts, will practice numerical problems on this.",
        ],
    },
    {
        "id": "sample-50",
        "title": "Sample 4 (DSA Graph BFS — Mastered/Understood)",
        "subject": "DSA",
        "topic": "Graph Traversal: BFS",
        "expected": "understood",
        "dataset_index": 50,
        "messages": [
            "explain BFS algorithm on graphs using adjacency list.",
            "why do we use a queue for BFS instead of a stack ?",
            "got it, queue ensures level-by-level traversal while stack would give DFS.",
            "in C++: queue<int> q; visited[src] = true; q.push(src); while(!q.empty())...",
            "makes total sense now! time complexity is O(V + E). Thank you!",
        ],
    },
    {
        "id": "sample-95",
        "title": "Sample 5 (ML Gradient Descent — Confused)",
        "subject": "ML",
        "topic": "Gradient Descent Optimization",
        "expected": "confused",
        "dataset_index": 95,
        "messages": [
            "how does gradient descent update weights in linear regression ?",
            "i am confused about learning rate alpha. What happens if it is too large or too small ?",
            "what is the difference between batch, stochastic, and mini-batch GD ? i am totally mixed up.",
            "why does my cost function diverge to infinity ? i don't understand what went wrong.",
            "can you explain again from scratch with simple numbers ? I am completely stuck.",
        ],
    },
]

# Create API router
router = APIRouter()


@router.get("/")
@router.get("/health")
def health_check(request: Request):
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "EduLens API",
        "version": "1.0.0",
        "runtime": "Vercel Serverless Function",
    }


@router.post("/predict")
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


@router.get("/samples")
def get_samples():
    """Returns pre-built sample learning dialogues matching Streamlit choices."""
    return {"samples": SAMPLE_CONVERSATIONS}


@router.get("/topics")
def get_topic_analytics():
    """Returns curriculum-level topic mastery breakdown across CS subjects."""
    global _cached_topics
    if _cached_topics is not None:
        return {"topics": _cached_topics}

    records = get_raw_dataset()
    if not records:
        return {"topics": [], "warning": "Primary dataset not found on server."}

    try:
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
                    "sessions": [],
                }

            topic_stats[key]["total_sessions"] += 1
            topic_stats[key]["total_turns"] += turns
            if label in topic_stats[key]:
                topic_stats[key][label] += 1
            
            # Store snippet for topic drill-down
            if len(topic_stats[key]["sessions"]) < 5:
                topic_stats[key]["sessions"].append({
                    "chat_id": rec.get("chat_id"),
                    "student_id": rec.get("student_id"),
                    "persona": rec.get("student_persona"),
                    "outcome": label,
                    "turns": turns,
                    "preview": " ".join(extract_student_messages(rec))[:180] + "...",
                })

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
                "sessions": item["sessions"],
            })

        summary.sort(key=lambda x: (x["subject"], x["topic"]))
        _cached_topics = summary
        return {"topics": _cached_topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate topic summary: {str(e)}")


@router.get("/model-info")
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


# Mount router on both root and /api prefixes so any Vercel rewrite configuration works!
app.include_router(router, prefix="")
app.include_router(router, prefix="/api")
