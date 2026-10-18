from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from system1_eval.schema import ModelPrediction, Scenario, is_prediction_correct


@dataclass
class CalibrationBin:
    bin_start: float
    bin_end: float
    avg_conf: float
    avg_acc: float
    count: int


def normalize_noul_confidence(prob_true: float) -> tuple[bool, float]:
    clamped = min(1.0, max(0.0, float(prob_true)))
    if clamped >= 0.5:
        return True, clamped
    return False, 1.0 - clamped


def compute_calibration_bins(
    outcomes: list[tuple[float, bool]], num_bins: int = 10
) -> list[CalibrationBin]:
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(num_bins)]
    for conf, is_correct in outcomes:
        c = min(1.0, max(0.0, float(conf)))
        idx = min(num_bins - 1, max(0, int(c * num_bins)))
        buckets[idx].append((c, bool(is_correct)))

    bins: list[CalibrationBin] = []
    for i in range(num_bins):
        b_start = round(i / num_bins, 4)
        b_end = round((i + 1) / num_bins, 4)
        items = buckets[i]
        if not items:
            bins.append(
                CalibrationBin(
                    bin_start=b_start,
                    bin_end=b_end,
                    avg_conf=round((b_start + b_end) / 2.0, 4),
                    avg_acc=0.0,
                    count=0,
                )
            )
        else:
            avg_c = sum(x[0] for x in items) / len(items)
            avg_a = sum(1.0 if x[1] else 0.0 for x in items) / len(items)
            bins.append(
                CalibrationBin(
                    bin_start=b_start,
                    bin_end=b_end,
                    avg_conf=round(avg_c, 6),
                    avg_acc=round(avg_a, 6),
                    count=len(items),
                )
            )
    return bins


def compute_ece(bins: list[CalibrationBin], total_count: int) -> float:
    if total_count <= 0:
        return 0.0
    ece = 0.0
    for b in bins:
        if b.count > 0:
            ece += (b.count / total_count) * abs(b.avg_acc - b.avg_conf)
    return round(ece, 6)


def compute_mce(bins: list[CalibrationBin]) -> float:
    non_empty = [abs(b.avg_acc - b.avg_conf) for b in bins if b.count > 0]
    if not non_empty:
        return 0.0
    return round(max(non_empty), 6)


def compute_brier_score(outcomes: list[tuple[float, bool]]) -> float:
    if not outcomes:
        return 0.0
    total = 0.0
    for conf, is_correct in outcomes:
        c = min(1.0, max(0.0, float(conf)))
        target = 1.0 if is_correct else 0.0
        total += (c - target) ** 2
    return round(total / len(outcomes), 6)


def compute_percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(float(v) for v in values)
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 3)
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return round(sorted_vals[low], 3)
    weight = rank - low
    interp = sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight
    return round(interp, 3)


def compute_summary_metrics(
    scenarios_by_id: dict[str, Scenario],
    predictions: list[ModelPrediction],
    cost_per_1m_input: float = 0.10,
    cost_per_1m_output: float = 0.40,
) -> dict[str, Any]:
    if not predictions:
        empty_bins = compute_calibration_bins([], num_bins=10)
        return {
            "count": 0,
            "accuracy": 0.0,
            "ece": 0.0,
            "mce": 0.0,
            "brier_score": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "mean_ms": 0.0,
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "cost_per_1k_usd": 0.0,
            "calibration_bins": [asdict(b) for b in empty_bins],
        }

    outcomes: list[tuple[float, bool]] = []
    latencies: list[float] = []
    in_tokens: list[int] = []
    out_tokens: list[int] = []

    for pred in predictions:
        sc = scenarios_by_id[pred.scenario_id]
        correct = is_prediction_correct(sc, pred)
        conf = min(1.0, max(0.0, float(pred.confidence)))
        outcomes.append((conf, correct))
        latencies.append(float(pred.latency_ms))
        in_tokens.append(int(pred.input_tokens))
        out_tokens.append(int(pred.output_tokens))

    n = len(outcomes)
    acc = sum(1.0 if c else 0.0 for _, c in outcomes) / n
    bins = compute_calibration_bins(outcomes, num_bins=10)
    ece = compute_ece(bins, total_count=n)
    mce = compute_mce(bins)
    brier = compute_brier_score(outcomes)

    avg_in = sum(in_tokens) / n
    avg_out = sum(out_tokens) / n
    cost_1k = ((avg_in * 1000.0) / 1_000_000.0) * cost_per_1m_input + (
        (avg_out * 1000.0) / 1_000_000.0
    ) * cost_per_1m_output

    return {
        "count": n,
        "accuracy": round(acc, 4),
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "brier_score": round(brier, 4),
        "p50_ms": compute_percentile(latencies, 50.0),
        "p95_ms": compute_percentile(latencies, 95.0),
        "mean_ms": round(sum(latencies) / n, 3),
        "avg_input_tokens": round(avg_in, 1),
        "avg_output_tokens": round(avg_out, 1),
        "cost_per_1k_usd": round(cost_1k, 5),
        "calibration_bins": [asdict(b) for b in bins],
    }
