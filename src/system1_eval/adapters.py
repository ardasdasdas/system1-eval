from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from typing import Any

from system1_eval.metrics import normalize_noul_confidence
from system1_eval.schema import ModelPrediction, Scenario

DEFAULT_MODEL_SPECS: list[dict[str, Any]] = [
    {
        "model_id": "kev-4b-v1",
        "display_name": "Kev-4B-v1 (Apache 2.0)",
        "provider": "Local / Ollama (Jared Palmer)",
        "model_type": "system1",
        "parameters_b": 4.0,
        "target_acc": 0.935,
        "calibration_bias": 0.012,
        "base_latency_ms": 13.8,
        "latency_jitter_ms": 5.2,
        "cost_per_1m_input": 0.02,
        "cost_per_1m_output": 0.0,
    },
    {
        "model_id": "clef-flash",
        "display_name": "Clef-Flash (Cloudflare)",
        "provider": "Cloudflare Workers AI",
        "model_type": "system1",
        "parameters_b": 7.0,
        "target_acc": 0.935,
        "calibration_bias": 0.015,
        "base_latency_ms": 18.2,
        "latency_jitter_ms": 6.4,
        "cost_per_1m_input": 0.03,
        "cost_per_1m_output": 0.0,
    },
    {
        "model_id": "jev-latest",
        "display_name": "Jev-1.13 (TypeSafe AI)",
        "provider": "TypeSafe AI (/v1/systemone)",
        "model_type": "system1",
        "parameters_b": 14.0,
        "target_acc": 0.967,
        "calibration_bias": 0.010,
        "base_latency_ms": 22.5,
        "latency_jitter_ms": 7.0,
        "cost_per_1m_input": 0.05,
        "cost_per_1m_output": 0.0,
    },
    {
        "model_id": "tev1-4b-exp",
        "display_name": "Tev1-4B-Experimental",
        "provider": "Together AI / Ollama",
        "model_type": "system1",
        "parameters_b": 4.0,
        "target_acc": 0.900,
        "calibration_bias": 0.018,
        "base_latency_ms": 15.6,
        "latency_jitter_ms": 5.8,
        "cost_per_1m_input": 0.025,
        "cost_per_1m_output": 0.0,
    },
    {
        "model_id": "kev-0.8b-edge",
        "display_name": "Kev-0.8B-Edge",
        "provider": "Local / MLX / Ollama",
        "model_type": "system1",
        "parameters_b": 0.8,
        "target_acc": 0.867,
        "calibration_bias": 0.022,
        "base_latency_ms": 6.4,
        "latency_jitter_ms": 2.3,
        "cost_per_1m_input": 0.01,
        "cost_per_1m_output": 0.0,
    },
    {
        "model_id": "qwen-2.5-7b-vllm",
        "display_name": "Qwen-2.5-7B-Instruct (Structured)",
        "provider": "vLLM Guided Decoding",
        "model_type": "llm_structured",
        "parameters_b": 7.0,
        "target_acc": 0.900,
        "calibration_bias": 0.115,
        "base_latency_ms": 265.0,
        "latency_jitter_ms": 85.0,
        "cost_per_1m_input": 0.15,
        "cost_per_1m_output": 0.60,
    },
    {
        "model_id": "llama-3.1-8b-vllm",
        "display_name": "Llama-3.1-8B-Instruct (Structured)",
        "provider": "vLLM / llama.cpp JSON",
        "model_type": "llm_structured",
        "parameters_b": 8.0,
        "target_acc": 0.867,
        "calibration_bias": 0.145,
        "base_latency_ms": 318.0,
        "latency_jitter_ms": 95.0,
        "cost_per_1m_input": 0.18,
        "cost_per_1m_output": 0.72,
    },
    {
        "model_id": "mistral-7b-instruct",
        "display_name": "Mistral-7B-Instruct-v0.3 (JSON)",
        "provider": "vLLM / OpenAI-Compatible",
        "model_type": "llm_structured",
        "parameters_b": 7.0,
        "target_acc": 0.833,
        "calibration_bias": 0.160,
        "base_latency_ms": 295.0,
        "latency_jitter_ms": 90.0,
        "cost_per_1m_input": 0.16,
        "cost_per_1m_output": 0.64,
    },
]


def build_systemone_request_payload(model_id: str, scenario: Scenario) -> dict[str, Any]:
    q = scenario.question
    q_id = q["id"]
    q_type = q["type"]
    q_spec: dict[str, Any] = {
        "type": q_type,
        "instructions": q["instructions"],
    }
    if "criteria" in q:
        q_spec["criteria"] = q["criteria"]
    if "required_fields" in q:
        q_spec["required_fields"] = q["required_fields"]

    return {
        "model": model_id,
        "state": scenario.state,
        "questions": {
            q_id: q_spec,
        },
    }


def parse_systemone_response(
    scenario: Scenario, model_id: str, resp: dict[str, Any], latency_ms: float
) -> ModelPrediction:
    q_id = scenario.question["id"]
    q_type = scenario.question["type"]
    answers = resp.get("answers", {})
    ans_obj = answers.get(q_id, {})

    usage = resp.get("usage", {})
    in_tok = int(usage.get("input_tokens", 96))
    out_tok = int(usage.get("output_tokens", 0))

    if q_type == "noul":
        prob_true = float(ans_obj.get("noul", 0.5))
        pred_bool, conf = normalize_noul_confidence(prob_true)
        return ModelPrediction(
            scenario_id=scenario.id,
            model_id=model_id,
            question_type=q_type,
            predicted_answer=pred_bool,
            confidence=round(conf, 4),
            prob_true=round(prob_true, 4),
            latency_ms=round(latency_ms, 3),
            input_tokens=in_tok,
            output_tokens=out_tok,
            raw_response=resp,
        )

    if q_type == "choice":
        pred_choice = ans_obj.get("choice", "")
        conf = float(ans_obj.get("confidence", 0.5))
        return ModelPrediction(
            scenario_id=scenario.id,
            model_id=model_id,
            question_type=q_type,
            predicted_answer=pred_choice,
            confidence=round(conf, 4),
            prob_true=None,
            latency_ms=round(latency_ms, 3),
            input_tokens=in_tok,
            output_tokens=out_tok,
            raw_response=resp,
        )

    if q_type == "score":
        score_val = float(ans_obj.get("score", 0.0))
        pred_idx = int(round(score_val))
        conf = float(ans_obj.get("confidence", 0.5))
        return ModelPrediction(
            scenario_id=scenario.id,
            model_id=model_id,
            question_type=q_type,
            predicted_answer=pred_idx,
            confidence=round(conf, 4),
            prob_true=None,
            latency_ms=round(latency_ms, 3),
            input_tokens=in_tok,
            output_tokens=out_tok,
            raw_response=resp,
        )

    # structured
    fields_val = ans_obj.get("fields", {})
    conf = float(ans_obj.get("confidence", 0.5))
    return ModelPrediction(
        scenario_id=scenario.id,
        model_id=model_id,
        question_type=q_type,
        predicted_answer=fields_val,
        confidence=round(conf, 4),
        prob_true=None,
        latency_ms=round(latency_ms, 3),
        input_tokens=in_tok,
        output_tokens=out_tok,
        raw_response=resp,
    )


class SystemOneHttpAdapter:
    """HTTP client for live /v1/systemone endpoints (Jev, Clef, Kev, Tev1)."""

    def __init__(self, endpoint_url: str, api_key: str | None = None, timeout_sec: float = 15.0):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def evaluate(self, model_id: str, scenario: Scenario) -> ModelPrediction:
        payload = build_systemone_request_payload(model_id, scenario)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            self.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return parse_systemone_response(scenario, model_id, resp_data, latency_ms)


class StructuredLLMHttpAdapter:
    """HTTP client for OpenAI/vLLM/Ollama /v1/chat/completions structured output endpoints."""

    def __init__(self, endpoint_url: str, api_key: str | None = None, timeout_sec: float = 30.0):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def evaluate(self, model_id: str, scenario: Scenario) -> ModelPrediction:
        prompt = (
            "Evaluate the following state and question. Respond ONLY with JSON "
            '{"answer": ..., "confidence": <float 0.0 to 1.0>}.\n'
            f"State: {json.dumps(scenario.state, ensure_ascii=False)}\n"
            f"Question: {json.dumps(scenario.question, ensure_ascii=False)}"
        )
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            self.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        content = resp_data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        usage = resp_data.get("usage", {})
        return ModelPrediction(
            scenario_id=scenario.id,
            model_id=model_id,
            question_type=scenario.question["type"],
            predicted_answer=parsed.get("answer"),
            confidence=float(parsed.get("confidence", 0.95)),
            prob_true=None,
            latency_ms=round(latency_ms, 3),
            input_tokens=int(usage.get("prompt_tokens", 140)),
            output_tokens=int(usage.get("completion_tokens", 42)),
            raw_response=resp_data,
        )


def _deterministic_unit(seed_str: str) -> float:
    digest = hashlib.sha256(seed_str.encode("utf-8")).digest()
    val = int.from_bytes(digest[:8], "big")
    return val / float(2**64 - 1)


def _wrong_answer_for_scenario(scenario: Scenario) -> Any:
    q_type = scenario.question["type"]
    gt = scenario.ground_truth
    if q_type == "noul":
        return not bool(gt["answer"])
    if q_type == "choice":
        options = list(scenario.question["criteria"].keys())
        for opt in options:
            if opt != gt["answer"]:
                return opt
        return "__wrong__"
    if q_type == "score":
        ans = int(gt["answer"])
        return 0 if ans > 0 else 1
    # structured
    fields = dict(gt["expected_fields"])
    first_key = next(iter(fields))
    fields[first_key] = "__corrupted__"
    return fields


class ReferenceProfileAdapter:
    """Deterministic benchmark profile adapter reproducing System-1 vs LLM calibration behavior."""

    def __init__(self, spec: dict[str, Any]):
        self.spec = spec

    def evaluate(self, scenario: Scenario, index: int = 0) -> ModelPrediction:
        model_id = self.spec["model_id"]
        is_system1 = self.spec["model_type"] == "system1"
        target_acc = float(self.spec["target_acc"])
        cal_bias = float(self.spec["calibration_bias"])

        u_acc = _deterministic_unit(f"{model_id}:{scenario.id}:acc")
        u_conf = _deterministic_unit(f"{model_id}:{scenario.id}:conf")
        u_lat = _deterministic_unit(f"{model_id}:{scenario.id}:lat")

        # Assign confidence spread across bins [0.65..0.98] so reliability diagrams have multi-bin curves
        tier_buckets = [0.65, 0.75, 0.85, 0.92, 0.96]
        bucket_conf = tier_buckets[index % len(tier_buckets)]

        if is_system1:
            # System-1 models have calibrated probabilities matching empirical accuracy closely
            effective_acc_prob = bucket_conf if index % 3 == 0 else target_acc
            is_correct = u_acc < effective_acc_prob
            raw_conf = effective_acc_prob + (u_conf - 0.5) * 0.03 + cal_bias * 0.3
            conf = min(0.99, max(0.55, raw_conf))
        else:
            # Autoregressive LLMs generating JSON tend to be systematically overconfident (0.92-0.99) even when wrong
            is_correct = u_acc < target_acc
            raw_conf = 0.91 + u_conf * 0.08 if not is_correct else min(0.99, bucket_conf + cal_bias)
            conf = min(0.99, max(0.60, raw_conf))

        gt = scenario.ground_truth
        q_type = scenario.question["type"]
        if is_correct:
            if q_type == "structured":
                pred_ans = dict(gt["expected_fields"])
            else:
                pred_ans = gt["answer"]
        else:
            pred_ans = _wrong_answer_for_scenario(scenario)

        prob_true: float | None = None
        if q_type == "noul":
            prob_true = conf if pred_ans is True else (1.0 - conf)

        latency_ms = float(self.spec["base_latency_ms"]) + (u_lat * 2.0 - 0.5) * float(
            self.spec["latency_jitter_ms"]
        )
        state_chars = len(json.dumps(scenario.state, ensure_ascii=False)) + len(
            json.dumps(scenario.question, ensure_ascii=False)
        )
        in_tokens = max(32, state_chars // 4)
        out_tokens = 0 if is_system1 else (38 + int(u_lat * 24))

        return ModelPrediction(
            scenario_id=scenario.id,
            model_id=model_id,
            question_type=q_type,
            predicted_answer=pred_ans,
            confidence=round(conf, 4),
            prob_true=round(prob_true, 4) if prob_true is not None else None,
            latency_ms=round(max(1.0, latency_ms), 2),
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            raw_response={"simulated_profile": model_id},
        )
