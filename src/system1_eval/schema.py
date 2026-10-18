from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any

VALID_TRACKS: tuple[str, ...] = (
    "agent_routing",
    "security_guardrail",
    "structured_triage",
)

VALID_QUESTION_TYPES: tuple[str, ...] = (
    "noul",
    "choice",
    "score",
    "structured",
)


class DatasetValidationError(ValueError):
    """Raised when a scenario or JSONL dataset file fails schema validation."""


@dataclass
class Scenario:
    id: str
    track: str
    title: str
    state: dict[str, Any]
    question: dict[str, Any]
    ground_truth: dict[str, Any]
    tags: list[str] = field(default_factory=list)


@dataclass
class ModelPrediction:
    scenario_id: str
    model_id: str
    question_type: str
    predicted_answer: Any
    confidence: float
    prob_true: float | None
    latency_ms: float
    input_tokens: int
    output_tokens: int
    raw_response: dict[str, Any] = field(default_factory=dict)


def validate_scenario(raw: dict[str, Any], source: str = "<memory>") -> Scenario:
    if not isinstance(raw, dict):
        raise DatasetValidationError(f"{source}: Scenario entry must be a JSON object")

    for req_key in ("id", "track", "title", "state", "question", "ground_truth"):
        if req_key not in raw:
            raise DatasetValidationError(f"{source}: Missing required key '{req_key}'")

    scenario_id = str(raw["id"]).strip()
    if not scenario_id:
        raise DatasetValidationError(f"{source}: 'id' must be a non-empty string")

    track = raw["track"]
    if track not in VALID_TRACKS:
        raise DatasetValidationError(
            f"{source} ({scenario_id}): Invalid track '{track}', expected one of {VALID_TRACKS}"
        )

    title = str(raw["title"]).strip()
    if not title:
        raise DatasetValidationError(f"{source} ({scenario_id}): 'title' must be non-empty")

    state = raw["state"]
    if not isinstance(state, dict) or not state:
        raise DatasetValidationError(f"{source} ({scenario_id}): 'state' must be a non-empty object")

    question = raw["question"]
    if not isinstance(question, dict):
        raise DatasetValidationError(f"{source} ({scenario_id}): 'question' must be an object")

    q_id = question.get("id")
    q_type = question.get("type")
    q_inst = question.get("instructions")
    if not q_id or not isinstance(q_id, str):
        raise DatasetValidationError(f"{source} ({scenario_id}): 'question.id' is required")
    if q_type not in VALID_QUESTION_TYPES:
        raise DatasetValidationError(
            f"{source} ({scenario_id}): Invalid question.type '{q_type}', expected {VALID_QUESTION_TYPES}"
        )
    if not q_inst or not isinstance(q_inst, str):
        raise DatasetValidationError(f"{source} ({scenario_id}): 'question.instructions' is required")

    ground_truth = raw["ground_truth"]
    if not isinstance(ground_truth, dict):
        raise DatasetValidationError(f"{source} ({scenario_id}): 'ground_truth' must be an object")

    if q_type == "noul":
        if not isinstance(ground_truth.get("answer"), bool):
            raise DatasetValidationError(
                f"{source} ({scenario_id}): 'noul' ground_truth.answer must be a boolean"
            )
    elif q_type == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise DatasetValidationError(
                f"{source} ({scenario_id}): 'choice' question.criteria must be a dict with >=2 options"
            )
        ans = ground_truth.get("answer")
        if ans not in criteria:
            raise DatasetValidationError(
                f"{source} ({scenario_id}): ground_truth.answer '{ans}' not found in question.criteria"
            )
    elif q_type == "score":
        criteria = question.get("criteria")
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise DatasetValidationError(
                f"{source} ({scenario_id}): 'score' question.criteria must be a list with >=2 levels"
            )
        ans = ground_truth.get("answer")
        if not isinstance(ans, int) or isinstance(ans, bool) or not (0 <= ans < len(criteria)):
            raise DatasetValidationError(
                f"{source} ({scenario_id}): 'score' ground_truth.answer must be an integer index in 0..{len(criteria) - 1}"
            )
    elif q_type == "structured":
        req_fields = question.get("required_fields")
        if not isinstance(req_fields, list) or not req_fields:
            raise DatasetValidationError(
                f"{source} ({scenario_id}): 'structured' question.required_fields must be a non-empty list"
            )
        expected_fields = ground_truth.get("expected_fields")
        if not isinstance(expected_fields, dict) or not expected_fields:
            raise DatasetValidationError(
                f"{source} ({scenario_id}): 'structured' ground_truth.expected_fields must be a non-empty dict"
            )
        for field_name in req_fields:
            if field_name not in expected_fields:
                raise DatasetValidationError(
                    f"{source} ({scenario_id}): required field '{field_name}' missing from ground_truth.expected_fields"
                )

    tags = raw.get("tags", [])
    if not isinstance(tags, list):
        raise DatasetValidationError(f"{source} ({scenario_id}): 'tags' must be a list")

    return Scenario(
        id=scenario_id,
        track=track,
        title=title,
        state=state,
        question=question,
        ground_truth=ground_truth,
        tags=[str(t) for t in tags],
    )


def load_jsonl_file(path: pathlib.Path | str) -> list[Scenario]:
    file_path = pathlib.Path(path)
    if not file_path.exists():
        raise DatasetValidationError(f"Dataset file not found: {file_path}")

    scenarios: list[Scenario] = []
    seen_ids: set[str] = set()
    lines = file_path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        source_loc = f"{file_path.name}:line {idx}"
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"{source_loc}: Invalid JSON ({exc.msg})") from exc

        scenario = validate_scenario(obj, source=source_loc)
        if scenario.id in seen_ids:
            raise DatasetValidationError(f"{source_loc}: Duplicate scenario id '{scenario.id}'")
        seen_ids.add(scenario.id)
        scenarios.append(scenario)

    return scenarios


def load_dataset_dir(dataset_dir: pathlib.Path | str) -> list[Scenario]:
    dir_path = pathlib.Path(dataset_dir)
    if not dir_path.exists() or not dir_path.is_dir():
        raise DatasetValidationError(f"Dataset directory not found: {dir_path}")

    files = sorted(dir_path.glob("*.jsonl"))
    if not files:
        raise DatasetValidationError(f"No .jsonl files found in {dir_path}")

    all_scenarios: list[Scenario] = []
    seen_ids: set[str] = set()
    for f in files:
        file_scenarios = load_jsonl_file(f)
        for sc in file_scenarios:
            if sc.id in seen_ids:
                raise DatasetValidationError(f"{f.name}: Duplicate scenario id '{sc.id}' across files")
            seen_ids.add(sc.id)
            all_scenarios.append(sc)

    return all_scenarios


def _scalars_match(expected: Any, actual: Any) -> bool:
    if expected is None or actual is None:
        return expected is actual

    if isinstance(expected, bool) or isinstance(actual, bool):
        if isinstance(expected, bool) and isinstance(actual, bool):
            return expected == actual
        if isinstance(expected, bool) and isinstance(actual, str):
            return actual.strip().lower() == ("true" if expected else "false")
        return False

    if isinstance(expected, (int, float)):
        try:
            actual_num = float(str(actual).strip())
            return math.isclose(float(expected), actual_num, rel_tol=1e-6, abs_tol=1e-6)
        except (ValueError, TypeError):
            return False

    return str(expected).strip().lower() == str(actual).strip().lower()


def is_prediction_correct(scenario: Scenario, prediction: ModelPrediction) -> bool:
    q_type = scenario.question["type"]
    gt = scenario.ground_truth
    pred_ans = prediction.predicted_answer

    if q_type == "noul":
        return isinstance(pred_ans, bool) and pred_ans == gt["answer"]

    if q_type == "choice":
        return str(pred_ans).strip() == str(gt["answer"]).strip()

    if q_type == "score":
        if isinstance(pred_ans, bool):
            return False
        try:
            return int(round(float(pred_ans))) == int(gt["answer"])
        except (ValueError, TypeError):
            return False

    if q_type == "structured":
        if not isinstance(pred_ans, dict):
            return False
        expected_fields: dict[str, Any] = gt["expected_fields"]
        for key, expected_val in expected_fields.items():
            if key not in pred_ans:
                return False
            if not _scalars_match(expected_val, pred_ans[key]):
                return False
        return True

    return False
