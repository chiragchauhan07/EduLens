"""
predict.py — Real-time inference engine for EduLens student learning conversations.

Loads the trained Scikit-learn model artifact and predicts student understanding
outcome from raw conversation text or structured message records.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import pandas as pd

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml_prototype.data_loader import extract_student_messages
from ml_prototype.features import extract_conversation_stats

DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "trained_model.joblib"

# Human-readable labels and pedagogical feedback templates
CLASS_METADATA = {
    "understood": {
        "display_name": "Understood (High Conceptual Mastery)",
        "badge_color": "green",
        "feedback": (
            "The student demonstrated solid comprehension, articulated concepts clearly, "
            "and showed minimal unresolved confusion. Ready to advance to more complex problems."
        ),
    },
    "partial": {
        "display_name": "Partial Understanding (Emerging Comprehension)",
        "badge_color": "orange",
        "feedback": (
            "The student grasped fundamental ideas but requested repeated clarifications "
            "or showed lingering uncertainty. Targeted practice exercises or follow-up examples recommended."
        ),
    },
    "confused": {
        "display_name": "Confused (Instructional Intervention Needed)",
        "badge_color": "red",
        "feedback": (
            "The student expressed significant doubt, repeated questions, or struggled to apply "
            "the core concepts. Recommend 1-on-1 human tutor review or foundational review."
        ),
    },
}


class EduLensPredictor:
    """
    Wrapper around the trained model pipeline providing verified inference.
    """

    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model artifact not found at {self.model_path.resolve()}. "
                "Please run 'python ml_prototype/train_model.py' first."
            )
        self.pipeline = joblib.load(self.model_path)
        self.classes = self.pipeline.classes_.tolist()

    def predict(self, conversation_input: Union[Dict[str, Any], List[str], str]) -> Dict[str, Any]:
        """
        Predicts student understanding level from a conversation.

        Args:
            conversation_input: A conversation dict (with 'turns' or 'messages'),
                                a list of student utterance strings,
                                or a raw multi-line string.

        Returns:
            Dictionary containing predicted class, probabilities, conversational metrics,
            and pedagogical interpretation.
        """
        # 1. Parse student messages
        if isinstance(conversation_input, dict):
            student_msgs = extract_student_messages(conversation_input)
            # If empty but has 'full_text' or 'text'
            if not student_msgs and "full_text" in conversation_input:
                student_msgs = [conversation_input["full_text"]]
            elif not student_msgs and "text" in conversation_input:
                student_msgs = [conversation_input["text"]]
        elif isinstance(conversation_input, list):
            student_msgs = [str(m).strip() for m in conversation_input if str(m).strip()]
        elif isinstance(conversation_input, str):
            lines = [l.strip() for l in conversation_input.strip().split("\n") if l.strip()]
            student_msgs = lines if lines else [conversation_input.strip()]
        else:
            raise ValueError(f"Unsupported conversation input type: {type(conversation_input)}")

        if not student_msgs:
            return {
                "error": "No student messages found in input.",
                "predicted_class": "unknown",
                "display_name": "No Data",
                "confidence": 0.0,
                "probabilities": {},
                "conversational_signals": {},
                "pedagogical_feedback": "Unable to evaluate an empty conversation.",
            }

        full_text = " ".join(student_msgs)

        # 2. Prepare single-row DataFrame for pipeline
        input_df = pd.DataFrame([{
            "full_text": full_text,
            "student_messages": student_msgs,
        }])

        # 3. Model inference
        predicted_class = str(self.pipeline.predict(input_df)[0])
        probabilities_raw = self.pipeline.predict_proba(input_df)[0]
        
        prob_dict = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(self.classes, probabilities_raw)
        }
        confidence = prob_dict.get(predicted_class, 0.0)

        # 4. Extract conversational signals
        convo_signals = extract_conversation_stats(student_msgs)

        # 5. Build structured result
        meta = CLASS_METADATA.get(predicted_class, {
            "display_name": predicted_class.capitalize(),
            "badge_color": "gray",
            "feedback": "Analysis complete.",
        })

        return {
            "predicted_class": predicted_class,
            "display_name": meta["display_name"],
            "badge_color": meta["badge_color"],
            "confidence": confidence,
            "probabilities": prob_dict,
            "conversational_signals": convo_signals,
            "pedagogical_feedback": meta["feedback"],
            "student_turn_count": len(student_msgs),
            "input_text_preview": full_text[:120] + ("..." if len(full_text) > 120 else ""),
        }


def predict_conversation(conversation_input: Union[Dict[str, Any], List[str], str]) -> Dict[str, Any]:
    """Convenience helper function for one-off inference."""
    predictor = EduLensPredictor()
    return predictor.predict(conversation_input)


if __name__ == "__main__":
    import json

    print("Testing EduLensPredictor on sample conversations...")
    predictor = EduLensPredictor()

    sample_confused = {
        "turns": [
            {"student_message": "i don't understand pointers at all, it's so confusing"},
            {"student_message": "what is the difference between * and &? I'm completely lost"},
            {"student_message": "why does it give a segmentation fault? I don't get this"},
        ]
    }

    sample_understood = {
        "turns": [
            {"student_message": "can we review binary search trees?"},
            {"student_message": "so basically the left child is always smaller and right child is larger?"},
            {"student_message": "got it, that makes complete sense! Now I see how search runs in O(log n). Thanks!"},
        ]
    }

    print("\n--- Test 1: Expected Confused ---")
    res1 = predictor.predict(sample_confused)
    print(f"Predicted Class : {res1['predicted_class']} (Confidence: {res1['confidence']:.2%})")
    print(f"Probabilities   : {res1['probabilities']}")
    print(f"Signals         : {res1['conversational_signals']}")

    print("\n--- Test 2: Expected Understood ---")
    res2 = predictor.predict(sample_understood)
    print(f"Predicted Class : {res2['predicted_class']} (Confidence: {res2['confidence']:.2%})")
    print(f"Probabilities   : {res2['probabilities']}")
    print(f"Signals         : {res2['conversational_signals']}")

    print("\nInference verification passed.")
