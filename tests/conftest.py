"""Shared pytest fixtures for synthetic light curves, configs, and datasets."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.utils.config import load_config


@pytest.fixture(scope="session")
def base_config():
    """Load the real project config.yaml — tests validate against actual
    project defaults, not a hand-rolled test config, so a change to
    config.yaml is immediately exercised by the test suite."""
    return load_config("config.yaml")


@pytest.fixture
def synthetic_transit_light_curve():
    """Build a synthetic light curve with a known, injected transit signal:
    slow stellar variability trend + periodic 1% transit dip + photon noise.

    Returns:
        dict with time, flux, period, epoch, duration_hours, and the
        boolean in_transit mask (ground truth), so tests can check
        recovered values against known injected parameters.
    """
    rng = np.random.default_rng(0)
    time = np.arange(0, 30, 1 / 48)  # 30 days, 30-min cadence
    period = 5.0
    epoch = 1.2
    duration_hours = 3.0
    duration_days = duration_hours / 24

    phase_true = np.mod(time - epoch + period / 2, period) - period / 2
    in_transit = np.abs(phase_true) < duration_days / 2

    flux = 1.0 + 0.02 * np.sin(2 * np.pi * time / 13.0)
    flux[in_transit] -= 0.01
    flux += rng.normal(0, 0.0005, size=time.shape)

    return {
        "time": time,
        "flux": flux,
        "period": period,
        "epoch": epoch,
        "duration_hours": duration_hours,
        "in_transit": in_transit,
    }


@pytest.fixture
def small_model_config(base_config):
    """A config with a smaller model, for fast CPU test execution — mirrors
    what `--smoke-test` uses in the real training script."""
    cfg = copy.deepcopy(base_config)
    cfg.data.global_view_bins = 2001
    cfg.data.local_view_bins = 201
    return cfg
