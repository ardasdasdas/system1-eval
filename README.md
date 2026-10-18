# ⚡ System1-Eval (DecisionArena)

Open-source benchmark suite, automated evaluation harness, and live calibration leaderboard for **System One Decision Models** (**Clef**, **Kev**, **Jev**, **Tev1** via `/v1/systemone`) and **Structured-Output Open-Weight LLMs** (**Llama 3**, **Qwen**, **Mistral**).

Unlike traditional text-generation benchmarks, **System1-Eval** evaluates:
1. **Decision Accuracy:** Exact choice, boolean (`noul`), ordinal risk (`score`), and structured JSON field extraction accuracy.
2. **Probability Calibration (`ECE`, `MCE`, `Brier Score`):** When a model claims **90% confidence (`0.90`)**, is it actually right **90% of the time**?
3. **Latency & Zero-Output-Token Efficiency:** `p50` / `p95` latency in milliseconds and output token overhead per 1,000 decisions.

---

## Architecture

- **A. Crowdsourced Scenario Pool (`datasets/*.jsonl`):**
  - `datasets/agent_routing.jsonl` — Coding agent & IDE tool routing decisions (`choice` across 10–12 tools).
  - `datasets/security_guardrail.jsonl` — Prompt injection, secret exfiltration, and operational risk scoring (`noul` and `score`).
  - `datasets/structured_triage.jsonl` — OCR-corrupted invoices, customs declarations, and server error log extraction (`structured`, `noul`, `choice`, `score`).
- **B. Automated Evaluation Harness (`src/system1_eval/`):**
  - Native `/v1/systemone` adapter (`SystemOneHttpAdapter`), OpenAI/vLLM/Ollama structured adapter (`StructuredLLMHttpAdapter`), and reproducible reference profiles (`ReferenceProfileAdapter`).
- **C. Live Dynamic Leaderboard (`system1-eval serve`):**
  - Interactive web UI with track filters, cost/latency sorting, and SVG **Reliability Diagrams (Calibration Curves)** ready for local use or Hugging Face Spaces (`port 7860`).

---

## Quick Start

```bash
# 1. Install in editable mode
pip install -e .

# 2. Validate crowdsourced datasets
python -m system1_eval.cli validate --dataset-dir datasets

# 3. Run the benchmark harness and print the leaderboard
python -m system1_eval.cli run --dataset-dir datasets --output results/leaderboard.json

# 4. Launch the interactive Web Leaderboard & Calibration Curve UI
python -m system1_eval.cli serve --port 7860
```
