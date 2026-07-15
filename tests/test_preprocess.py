"""Tests for src/data/preprocess.py — the scientifically load-bearing module.

These exercise detrending, phase-folding, and view generation against a
synthetic light curve with a known injected transit (see
tests/conftest.py::synthetic_transit_light_curve), so correctness is
checked against ground truth, not just "doesn't crash."
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.preprocess import (
    TransitParams,
    flatten_light_curve,
    generate_global_view,
    generate_local_view,
    phase_fold,
)


class TestTransitParams:
    def test_valid_params_construct(self):
        params = TransitParams(period_days=5.0, epoch_bkjd=1.2, duration_hours=3.0)
        assert params.period_days == 5.0

    @pytest.mark.parametrize("period", [0, -1.0, -100.0])
    def test_nonpositive_period_raises(self, period):
        with pytest.raises(ValueError, match="period_days"):
            TransitParams(period_days=period, epoch_bkjd=1.0, duration_hours=3.0)

    @pytest.mark.parametrize("duration", [0, -1.0])
    def test_nonpositive_duration_raises(self, duration):
        with pytest.raises(ValueError, match="duration_hours"):
            TransitParams(period_days=5.0, epoch_bkjd=1.0, duration_hours=duration)


class TestFlattenLightCurve:
    def test_removes_stellar_trend_and_preserves_transit_dip(self, synthetic_transit_light_curve):
        lc = synthetic_transit_light_curve
        flat = flatten_light_curve(lc["time"], lc["flux"], method="spline")

        out_of_transit_median = np.median(flat[~lc["in_transit"]])
        in_transit_median = np.median(flat[lc["in_transit"]])

        assert abs(out_of_transit_median - 1.0) < 0.01, "baseline should be ~1.0 after detrending"
        assert in_transit_median < out_of_transit_median, "transit dip must survive detrending"

    def test_savgol_method_also_works(self, synthetic_transit_light_curve):
        lc = synthetic_transit_light_curve
        flat = flatten_light_curve(lc["time"], lc["flux"], method="savgol")
        assert not np.isnan(flat).any()
        assert len(flat) == len(lc["time"])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="length mismatch"):
            flatten_light_curve(np.arange(10), np.arange(5))

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            flatten_light_curve(np.array([]), np.array([]))

    def test_unknown_method_raises(self, synthetic_transit_light_curve):
        lc = synthetic_transit_light_curve
        with pytest.raises(ValueError, match="Unknown detrending method"):
            flatten_light_curve(lc["time"], lc["flux"], method="not_a_real_method")


class TestPhaseFold:
    def test_phase_is_sorted_and_bounded(self, synthetic_transit_light_curve):
        lc = synthetic_transit_light_curve
        params = TransitParams(lc["period"], lc["epoch"], lc["duration_hours"])
        phase, flux = phase_fold(lc["time"], lc["flux"], params)

        assert np.all(np.diff(phase) >= 0), "phase must be sorted ascending"
        assert phase.min() >= -lc["period"] / 2 - 1e-9
        assert phase.max() <= lc["period"] / 2 + 1e-9
        assert len(phase) == len(flux) == len(lc["time"])

    def test_transit_centers_near_phase_zero(self, synthetic_transit_light_curve):
        """The whole point of phase-folding: after folding, the transit dip
        should be centered at phase ~0, regardless of how many orbits
        occurred across the observing baseline."""
        lc = synthetic_transit_light_curve
        params = TransitParams(lc["period"], lc["epoch"], lc["duration_hours"])
        flat = flatten_light_curve(lc["time"], lc["flux"])
        phase, folded_flux = phase_fold(lc["time"], flat, params)

        near_center_mask = np.abs(phase) < (lc["duration_hours"] / 24) / 2
        far_from_center_mask = np.abs(phase) > lc["period"] / 4

        assert np.median(folded_flux[near_center_mask]) < np.median(folded_flux[far_from_center_mask])


class TestViewGeneration:
    def test_global_view_shape_and_no_nans(self, synthetic_transit_light_curve):
        lc = synthetic_transit_light_curve
        params = TransitParams(lc["period"], lc["epoch"], lc["duration_hours"])
        flat = flatten_light_curve(lc["time"], lc["flux"])
        phase, folded_flux = phase_fold(lc["time"], flat, params)

        view = generate_global_view(phase, folded_flux, lc["period"], n_bins=201)
        assert view.shape == (201,)
        assert view.dtype == np.float32
        assert not np.isnan(view).any()

    def test_local_view_shows_dip_at_center(self, synthetic_transit_light_curve):
        lc = synthetic_transit_light_curve
        params = TransitParams(lc["period"], lc["epoch"], lc["duration_hours"])
        flat = flatten_light_curve(lc["time"], lc["flux"])
        phase, folded_flux = phase_fold(lc["time"], flat, params)

        duration_days = lc["duration_hours"] / 24
        view = generate_local_view(phase, folded_flux, duration_days, n_bins=51, num_durations=4)

        assert view.shape == (51,)
        center = len(view) // 2
        edge_region = np.concatenate([view[:5], view[-5:]])
        assert view[center] < edge_region.mean(), "local view center should dip relative to edges"

    def test_local_view_empty_coverage_raises(self):
        """A phase array with no points anywhere near the requested window
        should fail loudly, not silently return a garbage-filled view."""
        phase = np.array([10.0, 11.0, 12.0])  # nowhere near phase=0
        flux = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="empty"):
            generate_local_view(phase, flux, duration_days=0.01, n_bins=51, num_durations=1)
