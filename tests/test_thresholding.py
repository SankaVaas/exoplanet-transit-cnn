"""Tests for src/evaluation/thresholding.py.

Covers the exact scenario that motivated this module: a fixed 0.5
threshold producing 0 F1 despite a well-ranking model, and confirming
threshold optimization recovers a usable operating point.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.thresholding import find_optimal_threshold, metrics_at_threshold


@pytest.fixture
def low_shifted_predictions():
    """Probabilities that rank-order correctly but sit systematically below
    0.5 — mirrors the real finding from the end-to-end pipeline test."""
    rng = np.random.default_rng(0)
    n = 1000
    y_true = (rng.random(n) < 0.15).astype(int)
    y_prob = np.clip(y_true * 0.3 + rng.normal(0, 0.1, n) + 0.05, 0, 1)
    return y_true, y_prob


class TestFindOptimalThreshold:
    def test_recovers_usable_f1_despite_low_shifted_probs(self, low_shifted_predictions):
        y_true, y_prob = low_shifted_predictions
        default_metrics = metrics_at_threshold(y_true, y_prob, 0.5)

        best_threshold, best_f1 = find_optimal_threshold(y_true, y_prob, criterion="f1")
        optimal_metrics = metrics_at_threshold(y_true, y_prob, best_threshold)

        assert optimal_metrics["f1"] >= default_metrics["f1"]
        assert optimal_metrics["f1"] > 0

    def test_youden_criterion_runs_and_returns_valid_threshold(self, low_shifted_predictions):
        y_true, y_prob = low_shifted_predictions
        threshold, score = find_optimal_threshold(y_true, y_prob, criterion="youden")
        assert 0 <= threshold <= 1
        assert -1 <= score <= 1

    def test_single_class_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="one class"):
            find_optimal_threshold(np.zeros(10), rng.random(10))

    def test_unknown_criterion_raises(self, low_shifted_predictions):
        y_true, y_prob = low_shifted_predictions
        with pytest.raises(ValueError, match="Unknown criterion"):
            find_optimal_threshold(y_true, y_prob, criterion="bogus")


class TestMetricsAtThreshold:
    def test_returns_expected_keys(self, low_shifted_predictions):
        y_true, y_prob = low_shifted_predictions
        result = metrics_at_threshold(y_true, y_prob, 0.5)
        assert set(result.keys()) == {"threshold", "precision", "recall", "f1"}
