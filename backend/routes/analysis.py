"""
analysis.py — FastAPI routes for EduLens student understanding analyzer.

Endpoints:
- GET  /api/health
- GET  /api/model-info
- GET  /api/sample-conversation
- POST /api/analyze  (Supports both multipart/form-data file upload and direct application/json body)
"""

import json
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from backend.services.conversation_parser import parse_conversation_json, ParsedConversation
from backend.services.prediction import ModelRegistry, analyze_conversation_session

router = APIRouter(prefix="/api", tags=["Analysis"])


@router.get("/health")
def get_health() -> Dict[str, Any]:
    """Health check verifying API and model status."""
    registry = ModelRegistry.get_instance()
    return {
        "status": "healthy",
        "service": "EduLens",
        "version": "1.0.0",
        "model_loaded": registry.pipeline is not None,
    }


@router.get("/model-info")
def get_model_info() -> Dict[str, Any]:
    """Provides technical specifications of the deployed ML model and NLP pipeline."""
    registry = ModelRegistry.get_instance()
    meta = registry.metadata or {}
    metrics = meta.get("metrics", {})
    return {
        "model_name": meta.get("model_name", "Logistic Regression (Balanced)"),
        "model_type": "Scikit-Learn Multinomial Logistic Regression Pipeline",
        "algorithm": meta.get("algorithm", "logistic_regression"),
        "supported_understanding_classes": [
            "Understood",
            "Partially Understood",
            "Confused",
        ],
        "confidence_estimation_method": "Linguistic Marker Estimation (Hedging, Certainty, Self-Correction)",
        "features_dimension": meta.get("features_dim", 108),
        "feature_composition": {
            "tfidf_features": 100,
            "conversational_features": 8,
            "total_features": 108,
        },
        "performance_metrics": {
            "test_accuracy": metrics.get("test_accuracy", 0.9589),
            "macro_precision": metrics.get("macro_precision", 0.938),
            "macro_recall": metrics.get("macro_recall", 0.9429),
            "macro_f1": metrics.get("macro_f1", 0.9399),
            "cross_val_accuracy_mean": metrics.get("cv_accuracy_mean", 0.9378),
        },
        "status": "operational",
    }


@router.get("/sample-conversation")
def get_sample_conversation() -> Dict[str, Any]:
    """Returns a representative educational dialogue JSON payload for quick demo testing."""
    return {
        "topic": "Physics: Static vs. Kinetic Friction",
        "messages": [
            {
                "role": "assistant",
                "content": "Welcome to physics tutoring! What questions do you have about friction today?",
            },
            {
                "role": "student",
                "content": "Can we review friction? I want to make sure I understand the difference between static and kinetic.",
            },
            {
                "role": "assistant",
                "content": "Great topic! How would you describe static friction when you push an object that hasn't started moving yet?",
            },
            {
                "role": "student",
                "content": "So basically static friction adjusts to match whatever force I apply, until it reaches a maximum limit?",
            },
            {
                "role": "assistant",
                "content": "Spot on! And what happens once the object overcomes that maximum limit and begins sliding?",
            },
            {
                "role": "student",
                "content": "Then kinetic friction takes over, and it stays relatively constant regardless of how fast it slides! Now I see how normal force affects both.",
            },
            {
                "role": "assistant",
                "content": "Excellent formulation. That shows solid conceptual clarity.",
            },
            {
                "role": "student",
                "content": "Got it, that makes complete sense! Thank you so much for the explanation.",
            },
        ],
    }


@router.post("/analyze")
async def analyze_conversation(
    request: Request,
    file: Optional[UploadFile] = File(None),
    topic: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """
    Main analysis endpoint. Accepts:
    1. A multipart/form-data uploaded .json file with optional 'topic' form parameter.
    2. A direct application/json body payload.
    """
    raw_bytes = None
    override_topic = topic

    # Case A: File uploaded via multipart/form-data
    if file is not None:
        if not file.filename.lower().endswith(".json"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{file.filename}'. Only .json files are supported.",
            )
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded JSON file is empty.")

    # Case B: Direct JSON payload
    elif request.headers.get("content-type", "").startswith("application/json"):
        try:
            body_dict = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON syntax in request body: {str(e)}")

        if not body_dict:
            raise HTTPException(status_code=400, detail="Request body is empty.")

        if not override_topic and isinstance(body_dict, dict):
            override_topic = body_dict.get("topic") or body_dict.get("subject")

        raw_bytes = body_dict

    else:
        # Check if form data had raw text or if empty
        form_data = await request.form()
        raw_json_text = form_data.get("json_data") or form_data.get("text")
        if raw_json_text:
            raw_bytes = str(raw_json_text).encode("utf-8")
        else:
            raise HTTPException(
                status_code=400,
                detail="No conversation payload received. Please upload a JSON file or post JSON data.",
            )

    # Parse and Validate Conversation
    try:
        parsed: ParsedConversation = parse_conversation_json(raw_bytes, override_topic=override_topic)
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Failed to process conversation: {str(err)}")

    # Run ML and NLP analysis
    try:
        result = analyze_conversation_session(parsed)
        return result
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Inference execution error: {str(err)}")



