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
        try:
            pred = EduLensPredictor(model_path=model_path)
            # Verify inference works
            pred.predict(["test message to verify pipeline"])
            _predictor = pred
        except BaseException as err:
            print(f"Pre-trained artifact load failed ({err}), initializing model pipeline on the fly...")
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from ml_prototype.data_loader import load_primary_dataset
            from ml_prototype.features import build_feature_pipeline

            dataset_path = ROOT_DIR / "data" / "raw" / "edulens_chats_dataset.json"
            df = load_primary_dataset(dataset_path)
            X = df[["full_text", "student_messages"]]
            y = df["final_understanding"]

            pipeline = Pipeline([
                ("preprocessor", build_feature_pipeline()),
                ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, C=1.0)),
            ])
            pipeline.fit(X, y)

            pred = EduLensPredictor.__new__(EduLensPredictor)
            pred.model_path = model_path
            pred.pipeline = pipeline
            pred.classes = pipeline.classes_.tolist()
            _predictor = pred

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
    algorithm: Optional[str] = Field(
        None,
        description="Optional model algorithm: 'logistic_regression', 'naive_bayes', or 'linear_svm'",
    )
    explain: bool = Field(
        False,
        description="Whether to include local feature attribution drivers in the prediction response",
    )


class PipelineRequest(BaseModel):
    conversation: Optional[Union[str, List[str], Dict[str, Any]]] = Field(
        None,
        description="Text transcript, list of student messages, or structured session object with 'turns'/'messages'",
    )
    text: Optional[str] = Field(None, description="Plain text transcript of student messages")
    quiz_submission: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional quiz submission object with 'questions', 'answer_key', and 'student_answers'",
    )
    algorithm: Optional[str] = Field(
        "logistic_regression",
        description="Optional model algorithm: 'logistic_regression', 'naive_bayes', or 'linear_svm'",
    )
    explain: bool = Field(False, description="Whether to include local feature attribution drivers")
    topic: Optional[str] = Field(None, description="Optional topic override name")


class QuizEvaluationRequest(BaseModel):
    questions: Optional[List[Dict[str, Any]]] = None
    answer_key: Optional[Dict[Union[str, int], Any]] = None
    student_answers: Optional[Dict[Union[str, int], Any]] = None
    submission_payload: Optional[Dict[str, Any]] = None


class ConfidenceRequest(BaseModel):
    conversation: Optional[Union[str, List[str], Dict[str, Any]]] = None
    text: Optional[str] = None


class IngestRequest(BaseModel):
    student_id: Optional[str] = Field("student_001", description="Student identifier")
    messages: Optional[List[Dict[str, Any]]] = Field(None, description="List of turn dicts with 'role' and 'content'")
    conversation: Optional[Union[str, List[str], Dict[str, Any]]] = Field(None, description="Transcript, list, or session dict")
    text: Optional[str] = Field(None, description="Raw multiline transcript text")
    subject: Optional[str] = Field(None, description="Academic subject")
    topic: Optional[str] = Field(None, description="Learning topic")
    source_agent: Optional[str] = Field("chatgpt", description="Source agent, e.g. chatgpt, claude, cursor")
    quiz_submission: Optional[Dict[str, Any]] = Field(None, description="Optional quiz submission to evaluate concurrently")


class SubmitQuizRequest(BaseModel):
    student_id: Optional[str] = Field("student_001", description="Student identifier")
    session_id: Optional[str] = Field(None, description="Optional linked session ID")
    subject: Optional[str] = Field(None, description="Academic subject")
    topic: Optional[str] = Field(None, description="Learning topic")
    questions: Optional[List[Dict[str, Any]]] = Field(None, description="List of question dicts")
    answer_key: Optional[Dict[Union[str, int], Any]] = Field(None, description="Mapping question_id -> correct answer")
    student_answers: Optional[Dict[Union[str, int], Any]] = Field(None, description="Mapping question_id -> student answer")
    submission_payload: Optional[Dict[str, Any]] = Field(None, description="Alternative raw submission dict")


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


@router.get("/health")
def health_check(request: Request):
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "EduLens API",
        "version": "1.0.2",
        "runtime": "Vercel Serverless Function",
    }


@router.post("/predict")
def predict_understanding(req: PredictRequest):
    """
    Evaluates student dialogue and returns predicted comprehension level,
    probabilities, predicted quiz score, and extracted conversational signals.
    """
    input_data = req.conversation if req.conversation is not None else req.text
    if not input_data:
        raise HTTPException(status_code=400, detail="Either 'conversation' or 'text' must be provided.")

    try:
        if req.algorithm:
            predictor = EduLensPredictor(algorithm=req.algorithm)
        else:
            predictor = get_predictor()
        result = predictor.predict(input_data, explain=req.explain)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/pipeline")
def run_unified_pipeline(req: PipelineRequest):
    """
    Unified learning intelligence pipeline evaluating all 4 dimensions:
    Topic Understanding, Actual Quiz, Expressed Confidence, and Overall Topic Performance.
    """
    from ml_prototype.unified_pipeline import EduLensUnifiedPipeline

    input_data = req.conversation if req.conversation is not None else req.text
    if not input_data:
        raise HTTPException(status_code=400, detail="Either 'conversation' or 'text' must be provided.")

    try:
        pipeline = EduLensUnifiedPipeline(algorithm=req.algorithm or "logistic_regression")
        res = pipeline.process_session(
            session_input=input_data,
            quiz_submission=req.quiz_submission,
            explain=req.explain,
            topic_override=req.topic,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")


@router.post("/evaluate-quiz")
def evaluate_quiz(req: QuizEvaluationRequest):
    """
    Deterministically evaluates submitted student answers against an answer key.
    Calculates correct, incorrect, unanswered, and percentage score.
    """
    from ml_prototype.quiz_evaluator import evaluate_quiz_submission

    try:
        res = evaluate_quiz_submission(
            questions=req.questions,
            answer_key=req.answer_key,
            student_answers=req.student_answers,
            submission_payload=req.submission_payload,
        )
        if "error" in res and res.get("total_questions", 0) == 0:
            raise HTTPException(status_code=400, detail=res["error"])
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz evaluation failed: {str(e)}")


@router.post("/confidence")
def estimate_confidence(req: ConfidenceRequest):
    """
    Estimates student expressed learning confidence from conversational markers.
    """
    from ml_prototype.confidence import ExpressedConfidenceEstimator

    input_data = req.conversation if req.conversation is not None else req.text
    if not input_data:
        raise HTTPException(status_code=400, detail="Either 'conversation' or 'text' must be provided.")

    try:
        estimator = ExpressedConfidenceEstimator()
        return estimator.predict(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Confidence estimation failed: {str(e)}")


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


@router.get("/model-comparison")
def get_model_comparison():
    """Returns comparative benchmark metrics across Logistic Regression, Naive Bayes, and Linear SVM."""
    report_path = ROOT_DIR / "reports" / "model_comparison.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Model comparison report not found.")
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load comparison: {str(e)}")


@router.get("/quiz-model-info")
def get_quiz_model_info():
    """Returns evaluation benchmarks for post-session quiz score regressor."""
    report_path = ROOT_DIR / "reports" / "quiz_regressor_evaluation.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Quiz regressor report not found.")
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load quiz report: {str(e)}")


@router.get("/explainability")
def get_explainability_report():
    """Returns global model coefficient weights and feature importances."""
    report_path = ROOT_DIR / "reports" / "model_explainability.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Explainability report not found.")
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load explainability report: {str(e)}")


def resolve_user_context(request: Request, client_student_id: Optional[str] = None):
    """
    Enforces student isolation and authentication across EduLens endpoints.
    - If Authorization header is provided: validates Bearer token.
      Ensures the authenticated user cannot access or modify another student's data (403 Forbidden).
    - If in production mode: Bearer token is strictly mandatory (401 Unauthorized if missing).
    - If in development mode without header: permits local identity switching scoped to the requested student.
    """
    from backend.mcp_server.config import settings
    from backend.mcp_server.auth.authentication import token_manager, AuthenticationError, UserContext

    auth_header = request.headers.get("Authorization")
    if auth_header:
        try:
            user = token_manager.validate_token(auth_header)
        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=str(e))

        if client_student_id and client_student_id.strip() and client_student_id != user.user_id:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied: authenticated user '{user.user_id}' cannot access resources for student '{client_student_id}'."
            )
        return user

    if settings.is_production:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide an 'Authorization: Bearer <token>' header."
        )

    effective_id = client_student_id or settings.default_dev_user
    return UserContext(
        user_id=effective_id,
        username=f"Dev Student ({effective_id})",
        client_type="dev_switcher",
        is_authenticated=False,
    )


@router.get("/students")
def list_students(request: Request):
    """Returns list of distinct student identifiers present in storage."""
    from backend.mcp_server.config import settings
    from backend.mcp_server.services.storage_service import get_storage_service
    storage = get_storage_service()
    all_students = storage.get_all_students()
    auth_header = request.headers.get("Authorization")
    if auth_header:
        from backend.mcp_server.auth.authentication import token_manager
        try:
            user = token_manager.validate_token(auth_header)
            if settings.is_production:
                return {"students": [user.user_id]}
        except Exception:
            pass
    return {"students": all_students}


@router.get("/dashboard")
def get_dashboard(request: Request, student_id: Optional[str] = "student_001"):
    """
    Returns aggregated KPI metrics, understanding distribution, recent sessions,
    and areas of confusion for the student learning overview dashboard.
    """
    user = resolve_user_context(request, student_id)
    effective_id = user.user_id
    from backend.mcp_server.services.storage_service import get_storage_service
    storage = get_storage_service()
    summary = storage.get_student_summary(effective_id)
    total, recent_sessions = storage.list_sessions(effective_id, limit=6)
    return {
        "student_id": effective_id,
        "summary": {
            "total_sessions": summary.get("total_sessions", 0),
            "topics_studied_count": len(summary.get("topics_studied", [])),
            "average_understanding_score": summary.get("average_understanding_score", 0.0),
            "average_actual_quiz_score": summary.get("average_actual_quiz_score"),
            "average_confidence_score": summary.get("average_confidence_score", 0.0),
            "average_overall_score": summary.get("average_overall_score"),
        },
        "understanding_distribution": summary.get("understanding_distribution", {"understood": 0, "partial": 0, "confused": 0}),
        "areas_of_confusion": summary.get("areas_of_confusion", []),
        "topic_wise_breakdown": summary.get("topic_wise_breakdown", []),
        "trend": summary.get("learning_progress", []),
        "recent_sessions": recent_sessions,
    }


@router.get("/sessions")
def list_sessions(
    request: Request,
    student_id: Optional[str] = "student_001",
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    search: Optional[str] = None,
    class_filter: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
):
    """
    Returns filterable, searchable, paginated session list with understanding
    classifications, quiz statuses, and composite performance scores.
    """
    user = resolve_user_context(request, student_id)
    effective_id = user.user_id
    from backend.mcp_server.services.storage_service import get_storage_service
    storage = get_storage_service()
    total, items = storage.list_sessions(
        user_id=effective_id,
        subject=subject,
        topic=topic,
        search=search,
        class_filter=class_filter,
        limit=limit,
        offset=offset,
    )
    return {
        "total_count": total,
        "sessions": items,
        "limit": limit,
        "offset": offset,
        "student_id": effective_id,
    }


@router.get("/sessions/{session_id}")
def get_session_detail(session_id: str, request: Request, student_id: Optional[str] = "student_001"):
    """
    Returns full raw conversation dialogue, 4-dimension analytics,
    feature attributions, pedagogical recommendations, and linked quiz.
    """
    user = resolve_user_context(request, student_id)
    effective_id = user.user_id
    from backend.mcp_server.services.storage_service import get_storage_service
    storage = get_storage_service()
    raw = storage.get_raw_session(session_id, effective_id)
    if not raw:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found for student '{effective_id}'.")
    analytics = storage.get_analytics(session_id, effective_id)

    # Check for any linked quiz in quiz_submissions
    quizzes_count, quizzes = storage.list_quiz_submissions(effective_id, limit=50)
    linked_quiz = None
    for q in quizzes:
        if q.get("session_id") == session_id:
            linked_quiz = storage.get_quiz_submission(q["quiz_id"], effective_id)
            break

    messages = []
    if raw and "raw_json" in raw and "messages" in raw["raw_json"]:
        messages = raw["raw_json"]["messages"]

    return {
        "session_id": session_id,
        "user_id": effective_id,
        "subject": raw.get("subject"),
        "topic": raw.get("topic"),
        "source_agent": raw.get("source_agent"),
        "status": raw.get("status"),
        "turns_count": raw.get("turns_count"),
        "ingested_at": raw.get("ingested_at"),
        "messages": messages,
        "analytics": analytics.get("full_analytics_json") if analytics else None,
        "summary_metrics": {
            "predicted_class": analytics.get("predicted_class") if analytics else None,
            "understanding_score": analytics.get("understanding_score") if analytics else None,
            "actual_quiz_score": analytics.get("actual_quiz_score") if analytics else None,
            "expressed_confidence_level": analytics.get("expressed_confidence_level") if analytics else None,
            "expressed_confidence_score": analytics.get("expressed_confidence_score") if analytics else None,
            "overall_performance_score": analytics.get("overall_performance_score") if analytics else None,
            "performance_category": analytics.get("performance_category") if analytics else None,
            "discrepancy_detected": bool(analytics.get("discrepancy_detected")) if analytics else False,
            "discrepancy_details": analytics.get("discrepancy_details") if analytics else None,
            "pedagogical_feedback": analytics.get("pedagogical_feedback") if analytics else None,
        } if analytics else None,
        "linked_quiz": linked_quiz,
    }


@router.get("/student-performance")
def get_student_performance(request: Request, student_id: Optional[str] = "student_001"):
    """Returns longitudinal learning metrics and curriculum breakdown for the student."""
    user = resolve_user_context(request, student_id)
    effective_id = user.user_id
    from backend.mcp_server.services.storage_service import get_storage_service
    storage = get_storage_service()
    return storage.get_student_summary(effective_id)


@router.get("/topic-summary")
def get_topic_summary_endpoint(topic: str, request: Request, student_id: Optional[str] = "student_001"):
    """Returns topic-specific mastery, quiz performance, and progression trend."""
    user = resolve_user_context(request, student_id)
    effective_id = user.user_id
    from backend.mcp_server.services.storage_service import get_storage_service
    storage = get_storage_service()
    summary = storage.get_topic_summary(effective_id, topic)
    if not summary:
        return {
            "topic": topic,
            "student_id": effective_id,
            "total_sessions": 0,
            "understanding_distribution": {"understood": 0, "partial": 0, "confused": 0},
            "average_understanding_score": 0.0,
            "quiz_performance": {"average_score_percent": None, "quizzes_taken": 0},
            "expressed_confidence": {"average_confidence_score": 0.0},
            "overall_topic_performance": None,
            "historical_trend": [],
        }
    return summary


@router.get("/quizzes")
def list_quizzes(request: Request, student_id: Optional[str] = "student_001", limit: int = 20, offset: int = 0):
    """Returns list of evaluated quiz submissions for the student."""
    user = resolve_user_context(request, student_id)
    effective_id = user.user_id
    from backend.mcp_server.services.storage_service import get_storage_service
    storage = get_storage_service()
    count, items = storage.list_quiz_submissions(effective_id, limit=limit, offset=offset)
    return {"total_count": count, "quizzes": items, "student_id": effective_id}


@router.get("/quizzes/{quiz_id}")
def get_quiz_detail(quiz_id: str, request: Request, student_id: Optional[str] = "student_001"):
    """Returns full itemized question breakdown for a specific quiz submission."""
    user = resolve_user_context(request, student_id)
    effective_id = user.user_id
    from backend.mcp_server.services.storage_service import get_storage_service
    storage = get_storage_service()
    quiz = storage.get_quiz_submission(quiz_id, effective_id)
    if not quiz:
        raise HTTPException(status_code=404, detail=f"Quiz '{quiz_id}' not found for student '{effective_id}'.")
    return quiz


@router.post("/quizzes")
def submit_quiz_endpoint(req: SubmitQuizRequest, request: Request):
    """
    Deterministically evaluates submitted student answers against an answer key,
    persists the graded submission to database, and updates session overall score.
    """
    from backend.mcp_server.services.ingestion_service import IngestionService
    from backend.mcp_server.schemas.session_schema import QuizSubmissionRequest

    user = resolve_user_context(request, req.student_id)
    svc = IngestionService()

    questions = req.questions or []
    if not questions and req.submission_payload:
        questions = req.submission_payload.get("questions", [])

    if not questions:
        raise HTTPException(status_code=400, detail="No quiz questions provided for evaluation.")

    try:
        submission_req = QuizSubmissionRequest(
            questions=questions,
            session_id=req.session_id,
            subject=req.subject,
            topic=req.topic,
            answer_key=req.answer_key or (req.submission_payload.get("answer_key") if req.submission_payload else None),
            student_answers=req.student_answers or (req.submission_payload.get("student_answers") if req.submission_payload else None),
        )
        res = svc.submit_quiz_results(submission_req, user)
        return res.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz submission failed: {str(e)}")


@router.post("/ingest")
def ingest_session_endpoint(req: IngestRequest, request: Request):
    """
    Ingests an external AI learning session (from ChatGPT, Claude, etc.),
    persists raw conversation, executes unified ML analytics pipeline, and stores results.
    """
    from backend.mcp_server.services.ingestion_service import IngestionService
    from backend.mcp_server.schemas.session_schema import SessionIngestRequest, MessageItem

    user = resolve_user_context(request, req.student_id)
    svc = IngestionService()

    msg_items: List[MessageItem] = []
    if req.messages:
        for m in req.messages:
            msg_items.append(MessageItem(
                role=m.get("role", "user"),
                content=m.get("content", ""),
                timestamp=m.get("timestamp"),
            ))
    elif req.conversation:
        if isinstance(req.conversation, list):
            for item in req.conversation:
                msg_items.append(MessageItem(role="user", content=str(item)))
        elif isinstance(req.conversation, dict) and "messages" in req.conversation:
            for m in req.conversation["messages"]:
                msg_items.append(MessageItem(role=m.get("role", "user"), content=m.get("content", "")))
        elif isinstance(req.conversation, dict) and "turns" in req.conversation:
            for t in req.conversation["turns"]:
                if t.get("student_message"):
                    msg_items.append(MessageItem(role="user", content=t["student_message"]))
                if t.get("teacher_message"):
                    msg_items.append(MessageItem(role="assistant", content=t["teacher_message"]))
        elif isinstance(req.conversation, str):
            lines = [line.strip() for line in req.conversation.split("\n") if line.strip()]
            for line in lines:
                msg_items.append(MessageItem(role="user", content=line))
    elif req.text:
        lines = [line.strip() for line in req.text.split("\n") if line.strip()]
        for line in lines:
            msg_items.append(MessageItem(role="user", content=line))

    if not msg_items:
        raise HTTPException(status_code=400, detail="No student messages provided for ingestion.")

    try:
        ingest_req = SessionIngestRequest(
            student_id=user.user_id,
            subject=req.subject or "Unspecified Subject",
            topic=req.topic or "Unspecified Topic",
            source_agent=req.source_agent or "chatgpt",
            messages=msg_items,
            quiz_submission=req.quiz_submission,
        )
        res = svc.ingest_session(ingest_req, user)
        return res.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session ingestion failed: {str(e)}")


# Mount router on both root and /api prefixes so any Vercel rewrite configuration works!
app.include_router(router, prefix="")
app.include_router(router, prefix="/api")

# Mount static files directory for local development or full-stack execution
from fastapi.staticfiles import StaticFiles
public_dir = ROOT_DIR / "public"
if public_dir.exists():
    app.mount("/", StaticFiles(directory=str(public_dir), html=True), name="static")
