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
