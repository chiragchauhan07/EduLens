"""
test_teacher_demo.py — Automated test suite for EduLens Teacher Demonstration Dataset.

Verifies:
1. All 8 teacher demonstration JSON files exist in samples/teacher_demo/
2. All 8 files parse cleanly through parse_conversation_json()
3. All 8 files pass through the production ML inference and linguistic confidence pipeline
4. Expected learning behaviors and confidence signals are captured without errors
"""

import json
from pathlib import Path
import pytest

from backend.services.conversation_parser import parse_conversation_json
from backend.services.prediction import analyze_conversation_session, ModelRegistry

DEMO_DIR = Path(__file__).resolve().parent.parent / "samples" / "teacher_demo"

EXPECTED_FILES = [
    "01_strong_understanding.json",
    "02_partial_understanding.json",
    "03_conceptual_confusion.json",
    "04_low_confidence.json",
    "05_high_confidence.json",
    "06_learning_improvement.json",
    "07_misconception.json",
    "08_mathematical_reasoning.json",
]


@pytest.fixture(scope="module")
def model_registry():
    return ModelRegistry.get_instance()


def test_all_demo_files_exist():
    """Verify all 8 required demonstration files exist in samples/teacher_demo/."""
    assert DEMO_DIR.exists(), f"Directory not found: {DEMO_DIR}"
    for filename in EXPECTED_FILES:
        filepath = DEMO_DIR / filename
        assert filepath.exists(), f"Missing demonstration file: {filename}"
        assert filepath.stat().st_size > 0, f"File is empty: {filename}"


@pytest.mark.parametrize("filename", EXPECTED_FILES)
def test_demo_file_valid_json_and_schema(filename):
    """Verify each file contains valid JSON matching the EduLens schema."""
    filepath = DEMO_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "topic" in data, f"'topic' field missing in {filename}"
    assert isinstance(data["topic"], str) and len(data["topic"].strip()) > 0
    assert "messages" in data, f"'messages' field missing in {filename}"
    assert isinstance(data["messages"], list)
    assert 6 <= len(data["messages"]) <= 16, f"Expected 6-16 messages, got {len(data['messages'])} in {filename}"

    roles = {m.get("role") for m in data["messages"]}
    assert roles.issubset({"student", "assistant"}), f"Unexpected roles in {filename}: {roles}"
    assert "student" in roles, f"No student messages found in {filename}"
    assert "assistant" in roles, f"No assistant messages found in {filename}"


@pytest.mark.parametrize("filename", EXPECTED_FILES)
def test_demo_file_parsing(filename):
    """Verify parse_conversation_json processes each demonstration conversation without error."""
    filepath = DEMO_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    parsed = parse_conversation_json(raw_text)
    assert parsed.student_turn_count >= 3, f"Expected at least 3 student turns in {filename}"
    assert parsed.assistant_turn_count >= 3, f"Expected at least 3 assistant turns in {filename}"
    assert len(parsed.topic) > 0


@pytest.mark.parametrize("filename", EXPECTED_FILES)
def test_demo_file_inference_pipeline(filename, model_registry):
    """Verify each demonstration file runs through the complete ML pipeline and produces valid results."""
    filepath = DEMO_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    parsed = parse_conversation_json(raw_text)
    analysis = analyze_conversation_session(parsed)

    assert analysis["success"] is True
    assert analysis["topic"] == parsed.topic

    # Understanding prediction checks
    und = analysis["understanding"]
    assert und["label"] in ["Understood", "Partially Understood", "Confused"]
    assert 0.0 <= und["probability"] <= 1.0
    assert set(und["class_probabilities"].keys()) == {"Understood", "Partially Understood", "Confused"}
    assert pytest.approx(sum(und["class_probabilities"].values()), rel=1e-2) == 1.0

    # Expressed confidence checks
    conf = analysis["expressed_confidence"]
    assert conf["label"] in ["High Confidence", "Moderate Confidence", "Low Confidence"]
    assert 0.0 <= conf["score"] <= 100.0
    assert "signals" in conf


def test_specific_pedagogical_contrasts(model_registry):
    """Verify specific intended contrasts between demonstration files."""
    def run_file(fname):
        with open(DEMO_DIR / fname, "r", encoding="utf-8") as f:
            return analyze_conversation_session(parse_conversation_json(f.read()))

    res_strong = run_file("01_strong_understanding.json")
    res_confusion = run_file("03_conceptual_confusion.json")
    res_low_conf = run_file("04_low_confidence.json")
    res_high_conf = run_file("05_high_confidence.json")
    res_misconception = run_file("07_misconception.json")

    # 1. 01_strong_understanding should predict Understood with high probability
    assert res_strong["understanding"]["label"] == "Understood"
    assert res_strong["understanding"]["probability"] > 0.80

    # 2. 03_conceptual_confusion should predict Confused
    assert res_confusion["understanding"]["label"] == "Confused"

    # 3. 04_low_confidence should produce Low Confidence despite solid conceptual understanding
    assert res_low_conf["expressed_confidence"]["label"] == "Low Confidence"
    assert res_low_conf["understanding"]["label"] == "Understood"

    # 4. 05_high_confidence should produce High Confidence (score > 80)
    assert res_high_conf["expressed_confidence"]["label"] == "High Confidence"
    assert res_high_conf["expressed_confidence"]["score"] >= 80.0

    # 5. 07_misconception demonstrates confident but incorrect reasoning (High Confidence + Partially Understood)
    assert res_misconception["expressed_confidence"]["label"] == "High Confidence"
    assert res_misconception["understanding"]["label"] == "Partially Understood"
