"""Tests for src/evaluation/calibration.py.

The overconfidence/underconfidence recovery tests here are constructed
against a grid-search ground truth (see the debugging session that caught
a real LBFGS single-step-convergence bug during development) — a
regression on this file would be caught immediately by these tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.calibration import TemperatureScaler, expected_calibration_error


@pytest.fixture
def calibrated_logits_and_labels():
    """Perfectly-calibrated-by-construction logits: draw true probabilities,
    sample labels from exactly those probabilities, and compute the
    corresponding logit — so any known distortion factor `k` applied to
    these logits has a known correct recovery temperature."""
    rng = np.random.default_rng(0)
    n = 5000
    true_probs = rng.uniform(0.05, 0.95, n)
    labels = (rng.random(n) < true_probs).astype(np.float32)
    logits = np.log(true_probs / (1 - true_probs))
    return torch.tensor(logits, dtype=torch.float32), torch.tensor(labels)


class TestTemperatureScaler:
    def test_recovers_overconfidence_factor(self, calibrated_logits_and_labels):
        logits, labels = calibrated_logits_and_labels
        k = 3.0
        overconfident_logits = logits * k

        scaler = TemperatureScaler()
        fitted_T = scaler.fit(overconfident_logits, labels)

        # Grid-search ground truth for this exact fixture was ~2.90 (see
        # module docstring) — allow reasonable tolerance for RNG variation.
        assert abs(fitted_T - 2.90) < 0.3

    def test_recovers_underconfidence_factor(self, calibrated_logits_and_labels):
        logits, labels = calibrated_logits_and_labels
        underconfident_logits = logits * 0.3

        scaler = TemperatureScaler()
        fitted_T = scaler.fit(underconfident_logits, labels)
        assert fitted_T < 1.0

    def test_calibration_reduces_ece_for_overconfident_model(self, calibrated_logits_and_labels):
        logits, labels = calibrated_logits_and_labels
        overconfident_logits = logits * 3.0

        scaler = TemperatureScaler()
        scaler.fit(overconfident_logits, labels)

        raw_probs = torch.sigmoid(overconfident_logits).numpy()
        calibrated_probs = scaler.calibrate(overconfident_logits).numpy()

        ece_before = expected_calibration_error(labels.numpy(), raw_probs)
        ece_after = expected_calibration_error(labels.numpy(), calibrated_probs)
        assert ece_after < ece_before

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            TemperatureScaler().fit(torch.randn(5), torch.randn(6))

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            TemperatureScaler().fit(torch.tensor([]), torch.tensor([]))

    def test_temperature_stays_positive(self, calibrated_logits_and_labels):
        logits, labels = calibrated_logits_and_labels
        scaler = TemperatureScaler()
        fitted_T = scaler.fit(logits, labels)
        assert fitted_T > 0


class TestExpectedCalibrationError:
    def test_near_zero_for_well_calibrated_generator(self):
        rng = np.random.default_rng(0)
        probs = rng.random(5000)
        labels = (rng.random(5000) < probs).astype(int)
        ece = expected_calibration_error(labels, probs, n_bins=20)
        assert ece < 0.03

    def test_high_for_badly_miscalibrated_predictions(self):
        # Model predicts near-0 confidence for everything, but half the
        # labels are actually positive -- should show large miscalibration.
        labels = np.array([1] * 50 + [0] * 50)
        probs = np.full(100, 0.05)
        ece = expected_calibration_error(labels, probs)
        assert ece > 0.3
