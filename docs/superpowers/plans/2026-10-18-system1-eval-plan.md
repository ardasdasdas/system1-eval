# System1-Eval (DecisionArena) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build System1-Eval (DecisionArena), an open-source benchmark suite, evaluation harness, and live calibration leaderboard for System One Decision Models (`Clef`, `Kev`, `Jev`, `Tev1`) and structured-output LLMs (`Llama 3`, `Mistral`, `Qwen`).

**Architecture:** A modular Python 3.11+ package (`system1_eval`) structured around four decoupled layers: (1) Dataset Schema & Validator (`schema.py`) + Crowdsourced JSONL Tracks (`datasets/*.jsonl`), (2) Calibration & Performance Math Engine (`metrics.py`) computing 10-bin ECE, MCE, Brier score, Reliability Diagrams, and latency percentiles, (3) Multi-Backend Model Adapters & Runner (`adapters.py`, `runner.py`) supporting `/v1/systemone`, `/v1/chat/completions`, and deterministic reference profiles, and (4) CLI + Hugging Face Spaces-compatible Web Leaderboard (`web.py`, `cli.py`).

**Tech Stack:** Python 3.11+ standard library (`json`, `math`, `statistics`, `urllib.request`, `http.server`, `argparse`, `pathlib`) and `pytest` for testing.

**Spec:** `docs/superpowers/specs/2026-10-18-system1-eval-design.md`

## Global Constraints

- Runtime & Stack: Python 3.11+ (`system1-eval` package), minimal dependencies (standard library `http.server` / `urllib.request` / `json` / `math` / `statistics`, plus `pytest` for tests), ensuring zero installation friction on Windows, macOS, and Linux and instant Hugging Face Spaces compatibility.
- Deterministic Reproducibility: Given the same model responses, ECE, MCE, Brier score, and accuracy calculations must be 100% deterministic.
- Never require paid API keys to run the evaluation pipeline, tests, CLI, or interactive Leaderboard UI locally.
- Partition predictions into $M = 10$ equal-width confidence bins $[0.0, 0.1), \dots, [0.9, 1.0]$ for ECE, MCE, and Reliability Diagrams.

## Review Focus

1. **Confidence Exactly at `1.0` or `0.0` in 10-Bin ECE Partitioning:** A confidence value of `1.0` must land in the 10th bin (`index 9`, `[0.9, 1.0]`) rather than overflowing to a non-existent 11th bin (`index 10`), and confidences outside `[0.0, 1.0]` must be clamped or rejected cleanly.
2. **Binary `noul` False-Prediction Confidence Inversion:** When a model returns `noul = 0.10` (meaning 10% probability of `true`), its predicted class is `false` and its confidence in that predicted class is `1.0 - 0.10 = 0.90` (not `0.10`); failing to invert confidence for `< 0.5` predictions corrupts ECE.
3. **Partial vs. Exact Match in `structured` JSON Triage:** If a model returns extra whitespace, numeric strings vs. numbers, or missing keys on a `structured` invoice/log scenario, `evaluate_prediction` must handle missing keys without raising `KeyError` and compute exact normalized field match deterministically.
4. **Empty Prediction List or Empty Track Slice in Metrics:** Calling `compute_summary_metrics([])` on an empty slice must return zeroed metrics and 10 empty reliability bins rather than raising `ZeroDivisionError`.
5. **Malformed Community JSONL Lines in Dataset Validator:** Blank lines or invalid JSON lines in `datasets/*.jsonl` must report the exact file name and 1-based line number in `DatasetValidationError` rather than crashing with an unhandled traceback.

---

### Task 1: Project Scaffolding, Dataset Schema & Validator (`schema.py`)

**Files:**
- Create: `pyproject.toml`
- Create: `src/system1_eval/__init__.py`
- Create: `src/system1_eval/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: none
- Produces:
  - `VALID_TRACKS = ("agent_routing", "security_guardrail", "structured_triage")`
  - `VALID_QUESTION_TYPES = ("noul", "choice", "score", "structured")`
  - `class DatasetValidationError(ValueError)`
  - `@dataclass class Scenario`: `id: str`, `track: str`, `title: str`, `state: dict`, `question: dict`, `ground_truth: dict`, `tags: list[str]`
  - `@dataclass class ModelPrediction`: `scenario_id: str`, `model_id: str`, `question_type: str`, `predicted_answer: Any`, `confidence: float`, `prob_true: float | None`, `latency_ms: float`, `input_tokens: int`, `output_tokens: int`, `raw_response: dict`
  - `validate_scenario(raw: dict, source: str = "<memory>") -> Scenario`
  - `load_jsonl_file(path: Path) -> list[Scenario]`
  - `load_dataset_dir(dataset_dir: Path) -> list[Scenario]`
  - `is_prediction_correct(scenario: Scenario, prediction: ModelPrediction) -> bool`

- [ ] **Step 1: Write the failing tests in `tests/test_schema.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'system1_eval'`

- [ ] **Step 3: Create `pyproject.toml`, `src/system1_eval/__init__.py`, and implement `src/system1_eval/schema.py`**

Implement `validate_scenario`, `load_jsonl_file`, `load_dataset_dir`, and `is_prediction_correct` in `src/system1_eval/schema.py`. Normalize scalar values in `structured` field comparisons by trimming strings, comparing case-insensitively for strings, and comparing numeric strings (`"120.50"`) against floats (`120.5`) within `1e-6`. Install package in editable mode (`pip install -e .`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schema.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/system1_eval/__init__.py src/system1_eval/schema.py tests/test_schema.py
git commit -m "feat: add dataset schema validation and prediction correctness engine"
```

---

### Task 2: Mathematical Calibration & Performance Metrics Engine (`metrics.py`)

**Files:**
- Create: `src/system1_eval/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `Scenario`, `ModelPrediction`, `is_prediction_correct` from `system1_eval.schema`
- Produces:
  - `@dataclass class CalibrationBin`: `bin_start: float`, `bin_end: float`, `avg_conf: float`, `avg_acc: float`, `count: int`
  - `normalize_noul_confidence(prob_true: float) -> tuple[bool, float]` — returns `(predicted_bool, confidence_in_predicted_class)` where `confidence = prob_true if prob_true >= 0.5 else 1.0 - prob_true`
  - `compute_calibration_bins(outcomes: list[tuple[float, bool]], num_bins: int = 10) -> list[CalibrationBin]`
  - `compute_ece(bins: list[CalibrationBin], total_count: int) -> float`
  - `compute_mce(bins: list[CalibrationBin]) -> float`
  - `compute_brier_score(outcomes: list[tuple[float, bool]]) -> float`
  - `compute_percentile(values: list[float], pct: float) -> float`
  - `compute_summary_metrics(scenarios_by_id: dict[str, Scenario], predictions: list[ModelPrediction], cost_per_1m_input: float = 0.10, cost_per_1m_output: float = 0.40) -> dict`

- [ ] **Step 1: Write the failing tests in `tests/test_metrics.py`**

```python
import pytest
from system1_eval.metrics import (
    compute_brier_score,
    compute_calibration_bins,
    compute_ece,
    compute_mce,
    compute_summary_metrics,
    normalize_noul_confidence,
)


def test_normalize_noul_confidence_inverts_below_half():
    pred_bool, conf = normalize_noul_confidence(0.10)
    assert pred_bool is False
    assert conf == pytest.approx(0.90)

    pred_bool_hi, conf_hi = normalize_noul_confidence(0.85)
    assert pred_bool_hi is True
    assert conf_hi == pytest.approx(0.85)


def test_calibration_bins_and_ece_boundary_one():
    # 10 predictions with confidence 0.90 (and one 1.0), 9 correct and 1 wrong -> 90% accuracy vs ~0.91 conf
    outcomes = [(0.90, True)] * 8 + [(1.0, True), (0.90, False)]
    bins = compute_calibration_bins(outcomes, num_bins=10)
    assert len(bins) == 10
    assert bins[9].count == 10
    assert bins[9].avg_acc == pytest.approx(0.90)
    ece = compute_ece(bins, total_count=10)
    assert ece == pytest.approx(abs(0.90 - 0.91), abs=1e-6)
    mce = compute_mce(bins)
    assert mce == pytest.approx(abs(0.90 - 0.91), abs=1e-6)
    brier = compute_brier_score([(0.90, True), (0.90, False)])
    assert brier == pytest.approx(((0.10**2) + (0.90**2)) / 2.0)


def test_compute_summary_metrics_empty_slice():
    summary = compute_summary_metrics({}, [])
    assert summary["count"] == 0
    assert summary["accuracy"] == 0.0
    assert summary["ece"] == 0.0
    assert len(summary["calibration_bins"]) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'system1_eval.metrics'`

- [ ] **Step 3: Implement `src/system1_eval/metrics.py`**

Implement 10-bin calibration bucketing (`bin_idx = min(num_bins - 1, max(0, int(conf * num_bins)))`), ECE, MCE, Brier score (`(conf - (1.0 if is_correct else 0.0)) ** 2`), latency percentiles (`p50_ms`, `p95_ms`), and token cost per 1,000 decisions.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/system1_eval/metrics.py tests/test_metrics.py
git commit -m "feat: add ECE, MCE, Brier score, and reliability diagram metrics engine"
```

---

### Task 3: Crowdsourced Scenario Pool (`datasets/*.jsonl`) & CI Workflow

**Files:**
- Create: `datasets/agent_routing.jsonl`
- Create: `datasets/security_guardrail.jsonl`
- Create: `datasets/structured_triage.jsonl`
- Create: `.github/workflows/ci.yml`
- Test: `tests/test_datasets.py`

**Interfaces:**
- Consumes: `load_dataset_dir`, `VALID_TRACKS` from `system1_eval.schema`
- Produces: 30 validated benchmark scenarios (10 in `agent_routing.jsonl`, 10 in `security_guardrail.jsonl`, 10 in `structured_triage.jsonl`)

- [ ] **Step 1: Write the failing test in `tests/test_datasets.py`**

```python
import pathlib
from system1_eval.schema import VALID_TRACKS, load_dataset_dir


def test_crowdsourced_datasets_load_and_cover_all_tracks():
    root = pathlib.Path(__file__).resolve().parents[1] / "datasets"
    scenarios = load_dataset_dir(root)
    assert len(scenarios) >= 30
    by_track = {t: [s for s in scenarios if s.track == t] for t in VALID_TRACKS}
    for track, items in by_track.items():
        assert len(items) >= 10, f"Track {track} has fewer than 10 scenarios"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_datasets.py -v`
Expected: FAIL because `datasets/` does not exist yet

- [ ] **Step 3: Create `datasets/agent_routing.jsonl`, `datasets/security_guardrail.jsonl`, `datasets/structured_triage.jsonl`, and `.github/workflows/ci.yml`**

Populate 10 realistic scenarios per track:
- `agent_routing.jsonl`: 10 IDE/coding agent routing decisions (choosing the right tool among 10–12 tools for git, AST inspection, test execution, browser preview, database query, MCP tool call, etc.).
- `security_guardrail.jsonl`: 10 guardrail scenarios (`noul` and `score` questions testing prompt injection, `.env` credential exfiltration, destructive `rm -rf`, SQL injection vs. benign educational queries).
- `structured_triage.jsonl`: 10 messy invoice, receipt, and server error log extraction/triage scenarios (`structured` and `choice` questions).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_datasets.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add datasets/ .github/workflows/ci.yml tests/test_datasets.py
git commit -m "feat: add 30 crowdsourced benchmark scenarios across 3 tracks and CI workflow"
```

---

### Task 4: System One (`/v1/systemone`), Structured LLM & Reference Adapters (`adapters.py`, `runner.py`)

**Files:**
- Create: `src/system1_eval/adapters.py`
- Create: `src/system1_eval/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `Scenario`, `ModelPrediction`, `load_dataset_dir` from `system1_eval.schema`; `compute_summary_metrics`, `normalize_noul_confidence` from `system1_eval.metrics`
- Produces:
  - `build_systemone_request_payload(model_id: str, scenario: Scenario) -> dict` — formats `/v1/systemone` JSON `{model, state, questions}`
  - `parse_systemone_response(scenario: Scenario, model_id: str, resp: dict, latency_ms: float) -> ModelPrediction`
  - `class SystemOneHttpAdapter`: evaluates live `/v1/systemone` endpoints (`Jev`, `Clef`, `Kev`, `Tev1`)
  - `class StructuredLLMHttpAdapter`: evaluates live `/v1/chat/completions` endpoints (`vLLM`, `Ollama`, `OpenAI`)
  - `DEFAULT_MODEL_SPECS: list[dict]` — 7 benchmark models (`kev-4b-v1`, `clef-flash`, `tev1-4b-exp`, `jev-latest`, `kev-0.8b-edge`, `llama-3.1-8b-vllm`, `qwen-2.5-7b-vllm`)
  - `run_benchmark(dataset_dir: Path, output_path: Path | None = None, model_ids: list[str] | None = None) -> dict`
  - `format_terminal_report(leaderboard_data: dict) -> str`

- [ ] **Step 1: Write the failing tests in `tests/test_runner.py`**

```python
import pathlib
from system1_eval.adapters import build_systemone_request_payload, parse_systemone_response
from system1_eval.runner import format_terminal_report, run_benchmark
from system1_eval.schema import load_dataset_dir


def test_systemone_payload_and_response_parser():
    dataset_dir = pathlib.Path(__file__).resolve().parents[1] / "datasets"
    scenarios = load_dataset_dir(dataset_dir)
    choice_sc = next(s for s in scenarios if s.question["type"] == "choice")

    payload = build_systemone_request_payload("typesafe/jev-latest", choice_sc)
    assert payload["model"] == "typesafe/jev-latest"
    assert payload["state"] == choice_sc.state
    qid = choice_sc.question["id"]
    assert payload["questions"][qid]["type"] == "choice"

    fake_resp = {
        "model": "typesafe/jev-latest",
        "answers": {
            qid: {
                "type": "choice",
                "choice": choice_sc.ground_truth["answer"],
                "confidence": 0.91,
            }
        },
        "usage": {"input_tokens": 95, "output_tokens": 0},
    }
    pred = parse_systemone_response(choice_sc, "typesafe/jev-latest", fake_resp, latency_ms=14.2)
    assert pred.predicted_answer == choice_sc.ground_truth["answer"]
    assert pred.confidence == 0.91
    assert pred.output_tokens == 0


def test_run_benchmark_and_terminal_report(tmp_path: pathlib.Path):
    dataset_dir = pathlib.Path(__file__).resolve().parents[1] / "datasets"
    out_file = tmp_path / "leaderboard.json"
    result = run_benchmark(dataset_dir=dataset_dir, output_path=out_file)
    assert out_file.exists()
    assert len(result["models"]) >= 6

    kev = next(m for m in result["models"] if m["model_id"] == "kev-4b-v1")
    llama = next(m for m in result["models"] if m["model_id"] == "llama-3.1-8b-vllm")
    # System One models have zero output tokens, lower latency, and better calibration (lower ECE) than uncalibrated LLM JSON
    assert kev["overall"]["avg_output_tokens"] == 0
    assert kev["overall"]["ece"] < llama["overall"]["ece"]
    assert kev["overall"]["p50_ms"] < llama["overall"]["p50_ms"]

    report = format_terminal_report(result)
    assert "kev-4b-v1" in report
    assert "ECE" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'system1_eval.adapters'`

- [ ] **Step 3: Implement `src/system1_eval/adapters.py` and `src/system1_eval/runner.py`**

Implement `build_systemone_request_payload`, `parse_systemone_response`, `SystemOneHttpAdapter`, `StructuredLLMHttpAdapter`, and deterministic `ReferenceProfileAdapter` (using seeded hashing per `(model_id, scenario.id)` so each model produces realistic, reproducible predictions, confidences, latencies, and calibration curves across all 30 scenarios). Implement `run_benchmark` and `format_terminal_report`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_runner.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/system1_eval/adapters.py src/system1_eval/runner.py tests/test_runner.py
git commit -m "feat: add System One (/v1/systemone), LLM adapters, and benchmark runner"
```

---

### Task 5: Live Dynamic Leaderboard Web Server (`web.py`), CLI (`cli.py`) & Documentation

**Files:**
- Create: `src/system1_eval/web.py`
- Create: `src/system1_eval/cli.py`
- Create: `results/leaderboard.json`
- Create: `README.md`
- Test: `tests/test_cli_and_web.py`

**Interfaces:**
- Consumes: `load_dataset_dir` from `system1_eval.schema`; `run_benchmark`, `format_terminal_report` from `system1_eval.runner`
- Produces:
  - `create_app_server(host: str, port: int, dataset_dir: Path, results_path: Path) -> ThreadingHTTPServer`
  - `main(argv: list[str] | None = None) -> int` in `src/system1_eval/cli.py` supporting subcommands `validate`, `run`, `report`, `serve`

- [ ] **Step 1: Write the failing tests in `tests/test_cli_and_web.py`**

```python
import json
import pathlib
import threading
import urllib.request
from system1_eval.cli import main
from system1_eval.web import create_app_server


def test_cli_validate_run_and_report(tmp_path: pathlib.Path, capsys):
    dataset_dir = pathlib.Path(__file__).resolve().parents[1] / "datasets"
    out_path = tmp_path / "leaderboard.json"

    rc_val = main(["validate", "--dataset-dir", str(dataset_dir)])
    assert rc_val == 0

    rc_run = main(["run", "--dataset-dir", str(dataset_dir), "--output", str(out_path)])
    assert rc_run == 0
    assert out_path.exists()

    rc_rep = main(["report", "--results", str(out_path)])
    assert rc_rep == 0
    captured = capsys.readouterr().out
    assert "kev-4b-v1" in captured
    assert "clef-flash" in captured


def test_web_leaderboard_endpoints_and_ui(tmp_path: pathlib.Path):
    dataset_dir = pathlib.Path(__file__).resolve().parents[1] / "datasets"
    out_path = tmp_path / "leaderboard.json"
    main(["run", "--dataset-dir", str(dataset_dir), "--output", str(out_path)])

    server = create_app_server("127.0.0.1", 0, dataset_dir=dataset_dir, results_path=out_path)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            html = resp.read().decode("utf-8")
            assert "System1-Eval" in html
            assert "Reliability Diagram" in html

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/leaderboard") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert len(data["models"]) >= 6

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/scenarios") as resp:
            scenarios = json.loads(resp.read().decode("utf-8"))
            assert len(scenarios["scenarios"]) >= 30
    finally:
        server.shutdown()
        server.server_close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_and_web.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'system1_eval.cli'`

- [ ] **Step 3: Implement `src/system1_eval/web.py`, `src/system1_eval/cli.py`, generate `results/leaderboard.json`, and write `README.md`**

Implement the web dashboard in `src/system1_eval/web.py` (with interactive track filter tabs, model type/latency/calibration filters, dynamic SVG Reliability Diagram comparing selected models against the ideal $y=x$ calibration line, and Scenario Explorer tab) and CLI subcommands (`validate`, `run`, `report`, `serve`) in `src/system1_eval/cli.py`. Generate `results/leaderboard.json` and document usage + Hugging Face Spaces deployment in `README.md`.

- [ ] **Step 4: Run full test suite to verify all tests pass**

Run: `pytest -v`
Expected: PASS (all tests across `test_schema.py`, `test_metrics.py`, `test_datasets.py`, `test_runner.py`, and `test_cli_and_web.py` pass)

- [ ] **Step 5: Commit**

```bash
git add src/system1_eval/web.py src/system1_eval/cli.py results/leaderboard.json README.md tests/test_cli_and_web.py
git commit -m "feat: add live dynamic leaderboard web UI, CLI commands, and documentation"
```
