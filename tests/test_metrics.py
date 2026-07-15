"""Tests for src/evaluation/metrics.py."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.evaluation.metrics import (
    bootstrap_metric_ci,
    compute_point_metrics,
    full_evaluation_report,
)


@pytest.fixture
def synthetic_predictions():
    rng = np.random.default_rng(1)
    n = 500
    y_true = (rng.random(n) < 0.2).astype(int)
    y_prob = np.clip(y_true * 0.6 + rng.normal(0, 0.25, n) + 0.2, 0, 1)
    return y_true, y_prob


class TestComputePointMetrics:
    def test_metrics_in_valid_ranges(self, synthetic_predictions):
        y_true, y_prob = synthetic_predictions
        metrics = compute_point_metrics(y_true, y_prob)
        for key in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
            assert 0 <= metrics[key] <= 1

    def test_out_of_range_probabilities_raise(self, synthetic_predictions):
        y_true, y_prob = synthetic_predictions
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            compute_point_metrics(y_true, y_prob * 5 - 1)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_point_metrics(np.zeros(5), np.zeros(6))

    def test_single_class_returns_nan_auc(self):
        rng = np.random.default_rng(0)
        metrics = compute_point_metrics(np.zeros(20, dtype=int), rng.random(20))
        assert np.isnan(metrics["roc_auc"])
        assert np.isnan(metrics["pr_auc"])


class TestBootstrapCI:
    def test_point_estimate_within_ci(self, synthetic_predictions):
        y_true, y_prob = synthetic_predictions
        point, lo, hi = bootstrap_metric_ci(
            y_true, y_prob, average_precision_score, n_iterations=200, seed=1
        )
        assert lo <= point <= hi

    def test_ci_bounds_ordered(self, synthetic_predictions):
        y_true, y_prob = synthetic_predictions
        _, lo, hi = bootstrap_metric_ci(
            y_true, y_prob, average_precision_score, n_iterations=200, seed=1
        )
        assert lo <= hi


class TestFullReport:
    def test_all_metrics_present_with_valid_cis(self, synthetic_predictions):
        y_true, y_prob = synthetic_predictions
        report = full_evaluation_report(y_true, y_prob, n_bootstrap=200, seed=1)
        expected_keys = {"precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"}
        assert expected_keys.issubset(report.keys())
        for name, values in report.items():
            assert values["ci_lower"] <= values["point"] <= values["ci_upper"]
