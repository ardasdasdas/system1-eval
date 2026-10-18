import pathlib
import pytest
from system1_eval.schema import (
    DatasetValidationError,
    ModelPrediction,
    Scenario,
    is_prediction_correct,
    load_jsonl_file,
    validate_scenario,
)


def test_validate_choice_scenario_and_correctness():
    raw = {
        "id": "route_001",
        "track": "agent_routing",
        "title": "Pick git diff tool",
        "state": {"user_prompt": "unstaged changes"},
        "question": {
            "id": "selected_tool",
            "type": "choice",
            "instructions": "Pick tool",
            "criteria": {"git_diff": "diff", "git_log": "log"},
        },
        "ground_truth": {"answer": "git_diff"},
        "tags": ["git"],
    }
    sc = validate_scenario(raw)
    assert isinstance(sc, Scenario)
    pred = ModelPrediction(
        scenario_id="route_001",
        model_id="kev-4b-v1",
        question_type="choice",
        predicted_answer="git_diff",
        confidence=0.92,
        prob_true=None,
        latency_ms=12.5,
        input_tokens=80,
        output_tokens=0,
        raw_response={},
    )
    assert is_prediction_correct(sc, pred) is True


def test_structured_triage_missing_keys_and_normalization():
    raw = {
        "id": "struct_001",
        "track": "structured_triage",
        "title": "Extract invoice fields",
        "state": {"raw_text": "INV-99 Total: 120.50 TRY"},
        "question": {
            "id": "invoice_extract",
            "type": "structured",
            "instructions": "Extract fields",
            "required_fields": ["invoice_id", "total", "currency"],
        },
        "ground_truth": {
            "expected_fields": {"invoice_id": "INV-99", "total": 120.5, "currency": "TRY"}
        },
    }
    sc = validate_scenario(raw)
    pred_missing = ModelPrediction(
        scenario_id="struct_001",
        model_id="kev-4b-v1",
        question_type="structured",
        predicted_answer={"invoice_id": "INV-99"},
        confidence=0.85,
        prob_true=None,
        latency_ms=15.0,
        input_tokens=90,
        output_tokens=0,
        raw_response={},
    )
    assert is_prediction_correct(sc, pred_missing) is False

    pred_exact = ModelPrediction(
        scenario_id="struct_001",
        model_id="kev-4b-v1",
        question_type="structured",
        predicted_answer={"invoice_id": " inv-99 ", "total": "120.50", "currency": "try"},
        confidence=0.91,
        prob_true=None,
        latency_ms=15.0,
        input_tokens=90,
        output_tokens=0,
        raw_response={},
    )
    assert is_prediction_correct(sc, pred_exact) is True


def test_load_jsonl_reports_line_number_on_invalid_json(tmp_path: pathlib.Path):
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text('{"id": "ok"}\n{not_json}\n', encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="line 2"):
        load_jsonl_file(bad_file)
