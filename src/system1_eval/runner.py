from __future__ import annotations

import datetime
import json
import pathlib
from typing import Any

from system1_eval.adapters import DEFAULT_MODEL_SPECS, ReferenceProfileAdapter
from system1_eval.metrics import compute_summary_metrics
from system1_eval.schema import VALID_TRACKS, ModelPrediction, Scenario, is_prediction_correct, load_dataset_dir


def run_benchmark(
    dataset_dir: pathlib.Path | str,
    output_path: pathlib.Path | str | None = None,
    model_ids: list[str] | None = None,
) -> dict[str, Any]:
    scenarios = load_dataset_dir(dataset_dir)
    scenarios_by_id: dict[str, Scenario] = {s.id: s for s in scenarios}

    selected_specs = DEFAULT_MODEL_SPECS
    if model_ids:
        wanted = set(model_ids)
        selected_specs = [m for m in DEFAULT_MODEL_SPECS if m["model_id"] in wanted]

    models_results: list[dict[str, Any]] = []
    for spec in selected_specs:
        adapter = ReferenceProfileAdapter(spec)
        preds: list[ModelPrediction] = []
        per_scenario_details: list[dict[str, Any]] = []
        for idx, sc in enumerate(scenarios):
            pred = adapter.evaluate(sc, index=idx)
            preds.append(pred)
            per_scenario_details.append(
                {
                    "scenario_id": sc.id,
                    "track": sc.track,
                    "predicted_answer": pred.predicted_answer,
                    "confidence": pred.confidence,
                    "correct": is_prediction_correct(sc, pred),
                    "latency_ms": pred.latency_ms,
                    "output_tokens": pred.output_tokens,
                }
            )

        overall = compute_summary_metrics(
            scenarios_by_id,
            preds,
            cost_per_1m_input=spec["cost_per_1m_input"],
            cost_per_1m_output=spec["cost_per_1m_output"],
        )

        tracks_summary: dict[str, Any] = {}
        for track in VALID_TRACKS:
            track_preds = [p for p in preds if scenarios_by_id[p.scenario_id].track == track]
            tracks_summary[track] = compute_summary_metrics(
                scenarios_by_id,
                track_preds,
                cost_per_1m_input=spec["cost_per_1m_input"],
                cost_per_1m_output=spec["cost_per_1m_output"],
            )

        # Composite Arena Score: rewards high accuracy and penalizes poor calibration (ECE)
        arena_score = round((overall["accuracy"] * 100.0) - (overall["ece"] * 50.0), 2)

        models_results.append(
            {
                "model_id": spec["model_id"],
                "display_name": spec["display_name"],
                "provider": spec["provider"],
                "model_type": spec["model_type"],
                "parameters_b": spec["parameters_b"],
                "arena_score": arena_score,
                "overall": overall,
                "tracks": tracks_summary,
                "predictions": per_scenario_details,
            }
        )

    models_results.sort(key=lambda m: (-m["arena_score"], m["overall"]["ece"], m["overall"]["p50_ms"]))

    payload = {
        "benchmark": "System1-Eval (DecisionArena)",
        "version": "0.1.0",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_scenarios": len(scenarios),
        "tracks": list(VALID_TRACKS),
        "models": models_results,
    }

    if output_path is not None:
        out_file = pathlib.Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return payload


def format_terminal_report(leaderboard_data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 98)
    lines.append(
        f"System1-Eval (DecisionArena) Leaderboard -- Scenarios: {leaderboard_data.get('total_scenarios', 0)}"
    )
    lines.append("=" * 98)
    header = (
        f"{'Rank':<5} {'Model ID':<22} {'Type':<14} {'Arena':>7} {'Acc %':>8} "
        f"{'ECE(v)':>8} {'Brier(v)':>8} {'p50 ms':>8} {'Out Tok':>8}"
    )
    lines.append(header)
    lines.append("-" * 98)

    for rank, model in enumerate(leaderboard_data.get("models", []), start=1):
        ov = model["overall"]
        m_type = "System-1" if model["model_type"] == "system1" else "LLM-JSON"
        acc_pct = ov["accuracy"] * 100.0
        lines.append(
            f"{rank:<5} {model['model_id']:<22} {m_type:<14} {model['arena_score']:>7.2f} "
            f"{acc_pct:>7.1f}% {ov['ece']:>8.4f} {ov['brier_score']:>8.4f} "
            f"{ov['p50_ms']:>7.1f}ms {ov['avg_output_tokens']:>8.1f}"
        )

    lines.append("=" * 98)
    return "\n".join(lines)
