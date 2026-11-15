"""
features.py — NLP and conversational feature extraction for EduLens.

This module extracts interpretable, leakage-free features from student learning dialogue:
1. Structural Conversational Metrics: Turn count, word counts, questions asked, syntax signals.
2. Pedagogical Signals: Confusion indicators, clarification requests, comprehension affirmations.
3. Lexical / Vocabulary Features: TF-IDF n-grams representing conceptual vocabulary.
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# Regular expression patterns for conversational pedagogical signals
CONFUSION_PATTERN = re.compile(
    r"\b(confus\w*|lost|stuck|don'?t\s+understand|don'?t\s+get|not\s+getting|didn'?t\s+get|"
    r"doubt|hard|unclear|struggl\w*|what\s+does\s+that\s+mean|huh|i'?m\s+blank|wait\s+what)\b",
    re.IGNORECASE,
)

CLARIFICATION_PATTERN = re.compile(
    r"\b(explain\s+again|another\s+example|give\s+an\s+example|can\s+you\s+show|"
    r"clarif\w*|repeat|difference\s+between|why\s+does|how\s+come|step\s+by\s+step)\b",
    re.IGNORECASE,
)

UNDERSTANDING_PATTERN = re.compile(
    r"\b(got\s+it|makes?\s+sense|understood|now\s+i\s+get|i\s+see|clear\s+now|"
    r"so\s+basically|that\s+means|ah\s+okay|oh\s+okay|thank\s+you|thanks|perfect)\b",
    re.IGNORECASE,
)

CODE_SYNTAX_PATTERN = re.compile(
    r"(`{1,3}|[{};]|\b(def|class|return|int|float|void|print|import|for|while)\b|==|!=|->)",
    re.IGNORECASE,
)


def extract_conversation_stats(messages: List[str]) -> Dict[str, float]:
    """
    Computes summary conversational and pedagogical metrics from a list of student messages.
    
    Args:
        messages: List of student utterance strings.
        
    Returns:
        Dictionary of 8 numeric conversational signals.
    """
    if not messages:
        return {
            "num_turns": 0.0,
            "total_words": 0.0,
            "avg_turn_length": 0.0,
            "question_count": 0.0,
            "confusion_markers": 0.0,
            "clarification_markers": 0.0,
            "understanding_markers": 0.0,
            "code_indicators": 0.0,
        }
        
    full_text = " ".join(messages)
    words = full_text.split()
    total_words = len(words)
    num_turns = len(messages)
    avg_turn_length = total_words / max(1, num_turns)
    
    question_count = float(full_text.count("?"))
    confusion_markers = float(len(CONFUSION_PATTERN.findall(full_text)))
    clarification_markers = float(len(CLARIFICATION_PATTERN.findall(full_text)))
    understanding_markers = float(len(UNDERSTANDING_PATTERN.findall(full_text)))
    code_indicators = float(len(CODE_SYNTAX_PATTERN.findall(full_text)))
    
    return {
        "num_turns": float(num_turns),
        "total_words": float(total_words),
        "avg_turn_length": float(round(avg_turn_length, 2)),
        "question_count": question_count,
        "confusion_markers": confusion_markers,
        "clarification_markers": clarification_markers,
        "understanding_markers": understanding_markers,
        "code_indicators": code_indicators,
    }


class ConversationalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer that extracts dense conversational
    metrics from a DataFrame column of message lists. Supports optional feature subsetting.
    """
    
    DEFAULT_FEATURES = [
        "num_turns",
        "total_words",
        "avg_turn_length",
        "question_count",
        "confusion_markers",
        "clarification_markers",
        "understanding_markers",
        "code_indicators",
    ]
    
    def __init__(self, message_col: str = "student_messages", selected_features: Optional[List[str]] = None):
        self.message_col = message_col
        self.selected_features = selected_features
        self.feature_names_ = selected_features if selected_features is not None else list(self.DEFAULT_FEATURES)
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
        
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            messages_series = X[self.message_col]
        else:
            messages_series = X
            
        selected = getattr(self, "selected_features", None)
        features = []
        for msgs in messages_series:
            stats = extract_conversation_stats(msgs)
            if selected is not None:
                row = [float(stats.get(k, 0.0)) for k in selected]
            else:
                row = list(stats.values())
            features.append(row)
        return np.array(features, dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.array(getattr(self, "feature_names_", self.DEFAULT_FEATURES))


def build_feature_pipeline(
    scaler: str = "standard",
    max_tfidf_features: int = 100,
    include_tfidf: bool = True,
    include_convo: bool = True,
    convo_features: Optional[List[str]] = None,
    custom_stop_words: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Builds the Scikit-Learn feature union / column transformer.
    
    Supports:
    - Combined (TF-IDF + Conversational Signals)
    - TF-IDF Only (include_convo=False)
    - Conversational Only (include_tfidf=False)
    - Feature ablation (convo_features subset)
    - Shortcut vocabulary filtering (custom_stop_words)
    - Scaler selection ('standard' or 'minmax')
    """
    transformers = []

    if include_tfidf:
        if custom_stop_words:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            effective_stop_words = list(set(ENGLISH_STOP_WORDS).union(set(w.lower() for w in custom_stop_words)))
        else:
            effective_stop_words = "english"

        transformers.append((
            "text_tfidf",
            TfidfVectorizer(
                max_features=max_tfidf_features,
                stop_words=effective_stop_words,
                ngram_range=(1, 2),
                min_df=2,
            ),
            "full_text",
        ))

    if include_convo:
        if scaler == "minmax":
            scaler_transformer = MinMaxScaler()
        else:
            scaler_transformer = StandardScaler()

        convo_pipeline = Pipeline([
            ("extract", ConversationalFeatureExtractor(
                message_col="student_messages",
                selected_features=convo_features,
            )),
            ("scale", scaler_transformer),
        ])
        transformers.append((
            "convo_stats",
            convo_pipeline,
            ["student_messages"],
        ))

    if not transformers:
        raise ValueError("At least one feature transformer (TF-IDF or conversational) must be enabled.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    return preprocessor


if __name__ == "__main__":
    from ml_prototype.data_loader import load_primary_dataset
    df = load_primary_dataset()
    pipe = build_feature_pipeline()
    X_trans = pipe.fit_transform(df)
    print("Features extracted successfully.")
    print(f"Input shape: {df.shape}")
    print(f"Transformed feature matrix shape: {X_trans.shape}")
    print("Sample single-conversation stats:")
    print(extract_conversation_stats(df.iloc[0]["student_messages"]))
