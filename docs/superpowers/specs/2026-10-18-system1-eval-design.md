# System1-Eval (DecisionArena) Design Specification

## 1. Intent and Why

New-generation **System One Decision Models**—such as **Jev** (TypeSafe AI), **Clef** (Cloudflare), **Kev** (open-source Apache 2.0 family), and **Tev1** (Together AI)—operate fundamentally differently from autoregressive Large Language Models (LLMs). Instead of generating text token-by-token, they evaluate an input `state` against typed `questions` (`noul`, `choice`, `score`) via the `/v1/systemone` API in a single prefill pass, outputting calibrated probability distributions with zero output tokens and sub-20ms latencies.

When developers use Decision Models or open-weight LLMs (Llama 3, Mistral, Qwen) in production workflows—such as routing prompts to save tokens in Cursor/Codex/Antigravity, enforcing security guardrails, or triaging messy data—traditional text-generation benchmarks are useless. Developers need to know three things:
1. **Decision Accuracy:** Does the model pick the right tool, risk flag, or structured field?
2. **Probability Calibration:** When the model claims **90% confidence (`0.90`)**, is it actually right **90% of the time**? (Measured via Expected Calibration Error [ECE], Brier Score, and Reliability Diagrams).
3. **Latency & Token Efficiency:** What are the `p50`/`p95` latencies in milliseconds and how many output tokens (and dollars) are consumed per 1,000 decisions?

**System1-Eval (DecisionArena)** is an open-source benchmark platform consisting of:
- **A. Crowdsourced Scenario Pool (`datasets/`):** Community-contributed real-world scenarios across Agent Routing, Security Guardrails, and Structured Data Triage.
- **B. Automated Evaluation Harness (`system1_eval` CLI & CI):** A standardized runner that evaluates both `/v1/systemone` models (`Clef`, `Kev`, `Jev`, `Tev1`) and structured-decoding LLMs (`vLLM`, `llama.cpp`, `Ollama`, OpenAI-compatible endpoints) and computes accuracy, calibration curves, and latency percentiles.
- **C. Live Dynamic Leaderboard (`system1-eval serve` / HF Spaces Ready):** An interactive web interface with category/cost/latency filters and visual Reliability Diagrams (Calibration Curves).

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
1. **Standardized Crowdsourced Dataset Schema & Starter Pool:**
   - Provide a human-readable, Git-PR-friendly JSONL/JSON dataset format supporting `/v1/systemone` primitives (`noul`, `choice`, `score`) and structured JSON extraction assertions.
   - Ship with a curated starter dataset covering all three core tracks:
     - `agent_routing`: Selecting the right tool among 10+ candidate tools (`choice`).
     - `security_guardrail`: Detecting prompt injection / secret leakage (`noul` binary probability + `score` ordinal risk level).
     - `structured_triage`: Extracting and validating required fields from noisy/corrupted invoice and log texts (`structured` + `noul`/`choice`).
   - Provide a dataset validation command (`system1-eval validate`) and GitHub Actions workflow (`ci.yml`) so community PRs are automatically verified.
2. **Multi-Backend Automated Evaluation Harness:**
   - **System One Adapter (`/v1/systemone`):** Native HTTP client for Jev, Clef, Kev, and Tev1 endpoints taking `{model, state, questions}` and parsing `{answers, usage}`.
   - **OpenAI / vLLM / Ollama Structured LLM Adapter (`/v1/chat/completions`):** Evaluates open-weight LLMs (Llama 3, Mistral, Qwen) using structured JSON schema responses (with reported confidence / logprobs) so LLMs and System One models can be compared head-to-head on identical scenarios.
   - **Simulated / Replay Benchmark Profiles:** Built-in deterministic evaluation profiles for `kev-4b-v1`, `clef-flash`, `tev1-4b-exp`, `jev-latest`, `llama-3.1-8b-vllm`, `qwen-2.5-7b-vllm`, and `mistral-7b-instruct` so users can run the harness, generate leaderboard results, and inspect calibration curves offline immediately out of the box.
3. **Mathematical Calibration & Performance Scoring:**
   - **Expected Calibration Error (ECE):** Partition predictions into $M = 10$ equal-width confidence bins $[0.0, 0.1), \dots, [0.9, 1.0]$ and compute:
     $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
   - **Maximum Calibration Error (MCE):** $\max_{m} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$.
   - **Brier Score:** $\frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$, measuring probabilistic accuracy.
   - **Task Accuracy & Latency:** Top-1 Accuracy, Macro-F1 (for `noul`/`choice`), normalized MAE (for `score`), JSON Schema/Field Exact Match (for `structured_triage`), plus `p50_ms`, `p95_ms`, `input_tokens`, `output_tokens`, and `cost_per_1k_decisions_usd`.
4. **Interactive Live Leaderboard & Reliability Diagram UI:**
   - Serve an interactive web dashboard (`system1-eval serve`) with zero external frontend build step required (FastAPI/standard HTTP backend + clean responsive HTML/CSS/JS frontend, deployable directly to Hugging Face Spaces or local machine).
   - Features:
     - Multi-dimensional filtering & sorting (by Track: All / Agent Routing / Security Guardrail / Structured Triage; by Model Type: `System-1` vs `LLM Structured`; by max `p50` latency; by max ECE).
     - Interactive SVG **Reliability Diagram (Calibration Curve)** plotting Predicted Confidence ($x$-axis) vs. Empirical Accuracy ($y$-axis) against the ideal $y=x$ diagonal for any selected models.
     - Scenario Inspector tab allowing developers to browse the crowdsourced scenarios and trigger a live benchmark run from the UI or CLI.

### Non-Goals (v1)
1. **Model Training / Fine-Tuning:** This platform evaluates and benchmarks models; it does not train model weights.
2. **External Database Dependencies:** All scenarios (`datasets/*.jsonl`) and benchmark run artifacts (`results/*.json`) are stored as flat, Git-versioned files so the repository is 100% self-contained.

### Anti-Goals (Failure Conditions Even If Technically Working)
1. **Ignoring Confidence Calibration:** A benchmark that only reports accuracy % and ignores whether the model's probabilities are calibrated fails the core mission of System1-Eval.
2. **Requiring Paid API Keys to Run or Explore the Platform:** A developer cloning the repo must be able to run the full evaluation pipeline, tests, CLI, and interactive Leaderboard UI locally with zero paid API keys.

---

## 3. Constraints

- **Runtime & Stack:** Python 3.11+ (`system1-eval` package), minimal dependencies (standard library `http.server` / `urllib.request` / `json` / `math` / `statistics`, plus `pytest` for tests), ensuring zero installation friction on Windows, macOS, and Linux and instant Hugging Face Spaces compatibility.
- **Deterministic Reproducibility:** Given the same model responses, ECE, MCE, Brier score, and accuracy calculations must be 100% deterministic.

---

## 4. Architecture & Data Schemas (Builder's Calls)

*(The following technical details were settled as sensible defaults by the architect.)*

### 4.1 Dataset Scenario Schema (`datasets/<track>.jsonl`)
Each line in `datasets/agent_routing.jsonl`, `datasets/security_guardrail.jsonl`, and `datasets/structured_triage.jsonl` is a JSON object:
```json
{
  "id": "route_001",
  "track": "agent_routing",
  "title": "Select git diff inspection tool among 12 IDE tools",
  "state": {
    "user_prompt": "Show me only the unstaged changes in src/router.py before we commit",
    "workspace_has_git": true
  },
  "question": {
    "id": "selected_tool",
    "type": "choice",
    "instructions": "Which single tool should the coding agent invoke first?",
    "criteria": {
      "git_diff_unstaged": "View unstaged modifications in working tree for a file",
      "git_log": "View commit history",
      "git_commit": "Create a new commit",
      "read_file": "Read entire file contents",
      "search_code": "Regex grep across repository",
      "run_tests": "Execute test suite",
      "edit_file": "Modify lines in a file",
      "delete_file": "Remove file from disk",
      "list_dir": "List files in directory",
      "web_search": "Search external documentation",
      "ask_user": "Prompt user for clarification",
      "finish_task": "Complete task and return final answer"
    }
  },
  "ground_truth": {
    "answer": "git_diff_unstaged"
  },
  "tags": ["cursor", "antigravity", "tool-routing"]
}
```
- Supported `question.type` values:
  - `"noul"`: `ground_truth.answer` is `true` or `false`. Model prediction provides `probability` in $[0.0, 1.0]$ for `true` (so predicted class is `probability >= 0.5`, and confidence in the predicted class is `p` if `p >= 0.5` else `1 - p`).
  - `"choice"`: `ground_truth.answer` is the key in `criteria` (or for `structured_triage`, canonical normalized JSON string / choice key). Model prediction provides `choice` (top option) and `confidence` in $[0.0, 1.0]$ (probability assigned to the chosen option).
  - `"score"`: `ground_truth.answer` is an integer index `0..K-1` on the rubric scale, plus `ground_truth.binary_positive` (boolean, e.g., `score >= 2` is high-risk) so calibration (`ECE`/`Brier`) and exact/MAE accuracy can both be computed consistently.
  - `"structured"`: `ground_truth.expected_fields` is a dict of required extracted key-value pairs from a messy input document (e.g., invoice number, total, currency, tax rate), evaluated on field-level exact match and model-reported extraction confidence.

### 4.2 Model Result & Leaderboard Artifact Schema (`results/leaderboard.json`)
Each evaluated model entry contains:
- `model_id`, `display_name`, `provider`, `model_type` (`"system1"` or `"llm_structured"`), `parameters_b`
- Overall and per-track (`agent_routing`, `security_guardrail`, `structured_triage`) metrics:
  - `accuracy` (0.0 to 1.0)
  - `ece` (Expected Calibration Error, 0.0 to 1.0, lower is better)
  - `mce` (Maximum Calibration Error, 0.0 to 1.0, lower is better)
  - `brier_score` (0.0 to 1.0, lower is better)
  - `p50_ms`, `p95_ms`
  - `avg_input_tokens`, `avg_output_tokens`, `cost_per_1k_usd`
  - `calibration_bins`: list of 10 bin objects `{bin_start, bin_end, avg_conf, avg_acc, count}` for rendering Reliability Diagrams.

### 4.3 CLI Interface (`system1-eval`)
- `system1-eval validate [--dataset-dir datasets]` — Validates all scenario files against the schema and prints dataset statistics.
- `system1-eval run [--models all|...] [--dataset-dir datasets] [--output results/leaderboard.json]` — Runs the evaluation harness across the dataset and computes all accuracy, calibration (ECE, MCE, Brier, 10-bin reliability curve), latency, and token metrics.
- `system1-eval report [--results results/leaderboard.json]` — Prints a formatted terminal leaderboard table and calibration diagnostics.
- `system1-eval serve [--port 7860] [--results results/leaderboard.json]` — Launches the Live Dynamic Leaderboard & Calibration Explorer web server (default port `7860` for Hugging Face Spaces compatibility) with JSON API endpoints (`/api/leaderboard`, `/api/scenarios`, `/api/evaluate`).

---

## 5. What Is Left to the Builder

- Internal package file structure under `src/system1_eval/`.
- Specific realistic scenario content inside `datasets/agent_routing.jsonl`, `datasets/security_guardrail.jsonl`, and `datasets/structured_triage.jsonl` (at least 10 rich scenarios per track, 30+ total scenarios).
- Visual styling details of the web dashboard beyond the approved layout (leaderboard table on the left, interactive SVG reliability diagram on the right, track tabs, and scenario explorer).
