"""
Core preprocessing pipeline: raw light curve -> global/local view tensors.

This is the scientifically load-bearing module of the project. It follows
the view-folding design from Shallue & Vanderburg (2018):

  1. Flatten: remove long-term stellar variability trends (detrending),
     leaving only short-timescale features like transits.
  2. Phase-fold: wrap the time series on the known orbital period so all
     transits from all epochs stack on top of each other.
  3. Bin into two fixed-length views:
       - global view: the whole folded orbit at coarse resolution
         (default 2001 bins) — captures overall shape, secondary eclipses,
         out-of-transit variability.
       - local view: a zoomed window around the transit itself at finer
         relative resolution (default 201 bins, spanning
         local_view_num_durations x transit duration) — captures transit
         shape (V-shaped vs. U-shaped is a strong planet/binary
         discriminator).

Every function here operates on a single star/TCE at a time and is unit
tested in tests/test_preprocess.py with synthetic light curves of known
shape, since correctness here determines everything downstream.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate, signal

from src.utils.config import ConfigDict, load_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TransitParams:
    """Transit ephemeris for one KOI, used to phase-fold its light curve."""

    period_days: float
    epoch_bkjd: float
    duration_hours: float

    def __post_init__(self) -> None:
        if self.period_days <= 0:
            raise ValueError(f"period_days must be positive, got {self.period_days}")
        if self.duration_hours <= 0:
            raise ValueError(f"duration_hours must be positive, got {self.duration_hours}")


def flatten_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    method: str = "spline",
    break_factor: float = 1.5,
    savgol_window: int = 101,
    savgol_polyorder: int = 2,
) -> np.ndarray:
    """Remove long-term trends from a light curve, leaving transit signal.

    Args:
        time: cadence timestamps (days).
        flux: raw (or lightly cleaned) flux values, same length as time.
        method: "spline" fits a piecewise cubic spline through the data
            with breakpoints spaced `break_factor` x median cadence gap
            apart, then divides it out. "savgol" applies a Savitzky-Golay
            filter as a faster, simpler alternative.
        break_factor: spline breakpoint spacing multiplier (spline only).
        savgol_window: window length for Savitzky-Golay (savgol only,
            must be odd).
        savgol_polyorder: polynomial order for Savitzky-Golay (savgol only).

    Returns:
        Flattened (detrended) flux, normalized so the out-of-transit
        baseline is ~1.0 — i.e. flux / trend.

    Raises:
        ValueError: on empty input, mismatched lengths, or an unknown
            method string — fails loudly rather than silently returning
            unflattened data.
    """
    if len(time) != len(flux):
        raise ValueError(f"time and flux length mismatch: {len(time)} vs {len(flux)}")
    if len(time) == 0:
        raise ValueError("Cannot flatten an empty light curve.")

    if method == "spline":
        # Breakpoints spaced by the median cadence gap x break_factor —
        # roughly matches the "spline through the data at ~1.5x the
        # typical gap" heuristic used in Kepler flattening pipelines.
        median_gap = np.median(np.diff(np.sort(time)))
        n_breaks = max(int((time.max() - time.min()) / (median_gap * break_factor * 100)), 4)
        breakpoints = np.linspace(time.min(), time.max(), n_breaks)[1:-1]
        try:
            spline = interpolate.LSQUnivariateSpline(time, flux, t=breakpoints, k=3)
            trend = spline(time)
        except Exception:
            # Degenerate cases (too few points, duplicate knots) — fall
            # back to savgol rather than crashing the whole batch.
            logger.warning("Spline fit failed, falling back to Savitzky-Golay.")
            return flatten_light_curve(time, flux, method="savgol")
    elif method == "savgol":
        window = min(savgol_window, len(flux) - (1 - len(flux) % 2))
        window = max(window - (window % 2 == 0), savgol_polyorder + 1)
        trend = signal.savgol_filter(flux, window_length=window, polyorder=savgol_polyorder)
    else:
        raise ValueError(f"Unknown detrending method: '{method}'. Use 'spline' or 'savgol'.")

    trend = np.where(trend == 0, np.median(flux), trend)  # guard divide-by-zero
    return flux / trend


def phase_fold(
    time: np.ndarray, flux: np.ndarray, params: TransitParams
) -> tuple[np.ndarray, np.ndarray]:
    """Fold a light curve on the transit period, centering the transit at phase 0.

    Args:
        time: cadence timestamps (days), same convention as `params.epoch_bkjd`.
        flux: (flattened) flux values.
        params: ephemeris to fold on.

    Returns:
        (phase, flux) sorted by phase ascending, where phase is in
        days-from-transit-center, wrapped to [-period/2, +period/2].
    """
    phase = np.mod(time - params.epoch_bkjd + params.period_days / 2, params.period_days)
    phase -= params.period_days / 2
    order = np.argsort(phase)
    return phase[order], flux[order]


def _bin_phase_curve(
    phase: np.ndarray, flux: np.ndarray, bin_edges: np.ndarray
) -> np.ndarray:
    """Median-bin a phase-folded curve into fixed bins, linearly interpolating
    any empty bins so the output is always a complete fixed-length vector
    (required for batching into a fixed-size CNN input)."""
    n_bins = len(bin_edges) - 1
    binned = np.full(n_bins, np.nan)
    bin_idx = np.digitize(phase, bin_edges) - 1
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            binned[b] = np.median(flux[mask])

    nan_mask = np.isnan(binned)
    if nan_mask.all():
        raise ValueError("All bins empty — insufficient phase coverage for this view.")
    if nan_mask.any():
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        binned[nan_mask] = np.interp(
            bin_centers[nan_mask], bin_centers[~nan_mask], binned[~nan_mask]
        )
    return binned


def generate_global_view(
    phase: np.ndarray, flux: np.ndarray, period_days: float, n_bins: int = 2001
) -> np.ndarray:
    """Bin the full phase-folded orbit into a fixed-length vector.

    Args:
        phase: folded phase (days from transit center), from `phase_fold`.
        flux: corresponding flattened flux.
        period_days: orbital period, defines the bin range [-P/2, P/2].
        n_bins: output vector length (config: data.global_view_bins).

    Returns:
        float32 array of shape (n_bins,), normalized to zero median and
        unit variance of the out-of-transit baseline.
    """
    bin_edges = np.linspace(-period_days / 2, period_days / 2, n_bins + 1)
    binned = _bin_phase_curve(phase, flux, bin_edges)
    return _normalize_view(binned)


def generate_local_view(
    phase: np.ndarray,
    flux: np.ndarray,
    duration_days: float,
    n_bins: int = 201,
    num_durations: float = 4.0,
) -> np.ndarray:
    """Bin a zoomed window around the transit into a fixed-length vector.

    Args:
        phase: folded phase (days from transit center).
        flux: corresponding flattened flux.
        duration_days: transit duration in days.
        n_bins: output vector length (config: data.local_view_bins).
        num_durations: window half-width in units of transit duration —
            a window of num_durations x duration_days is centered on phase 0.

    Returns:
        float32 array of shape (n_bins,), same normalization as the global view.
    """
    half_window = (num_durations * duration_days) / 2
    bin_edges = np.linspace(-half_window, half_window, n_bins + 1)
    binned = _bin_phase_curve(phase, flux, bin_edges)
    return _normalize_view(binned)


def _normalize_view(view: np.ndarray) -> np.ndarray:
    """Normalize a binned view to zero median, scaled by out-of-transit std.

    Using the median (not mean) as the baseline estimator is deliberate:
    the transit itself is a one-sided outlier (flux only ever dips, never
    spikes for a real transit), so the median is a robust estimate of the
    true out-of-transit baseline even with the transit included in the window.
    """
    baseline = np.median(view)
    spread = np.std(view) if np.std(view) > 0 else 1.0
    return ((view - baseline) / spread).astype(np.float32)


def process_single_koi(
    light_curve_path: Path, params: TransitParams, cfg: ConfigDict
) -> dict[str, np.ndarray] | None:
    """Full pipeline for one KOI: load -> flatten -> fold -> global/local views.

    Args:
        light_curve_path: path to the raw light curve CSV (time, flux, flux_err).
        params: transit ephemeris for this KOI.
        cfg: full config (uses cfg.data.* for bin counts, detrending method).

    Returns:
        dict with "global_view" and "local_view" float32 arrays, or None
        if the light curve file is missing/unusable (logged as a warning
        so it's visible in the pipeline run log, not silently dropped).
    """
    if not light_curve_path.exists():
        logger.warning(f"Missing light curve file: {light_curve_path}")
        return None

    try:
        df = pd.read_csv(light_curve_path)
        time = df["time"].to_numpy(dtype=np.float64)
        flux = df["flux"].to_numpy(dtype=np.float64)

        valid = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[valid], flux[valid]
        if len(time) < 50:
            logger.warning(f"{light_curve_path.name}: too few valid cadences ({len(time)}).")
            return None

        flat_flux = flatten_light_curve(
            time, flux, method=cfg.data.detrending_method, break_factor=cfg.data.spline_break_factor
        )
        phase, folded_flux = phase_fold(time, flat_flux, params)

        global_view = generate_global_view(
            phase, folded_flux, params.period_days, n_bins=cfg.data.global_view_bins
        )
        local_view = generate_local_view(
            phase,
            folded_flux,
            duration_days=params.duration_hours / 24.0,
            n_bins=cfg.data.local_view_bins,
            num_durations=cfg.data.local_view_num_durations,
        )
        return {"global_view": global_view, "local_view": local_view}

    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to process {light_curve_path.name}: {exc}")
        return None


def build_processed_dataset(cfg: ConfigDict) -> pd.DataFrame:
    """Run the full preprocessing pipeline over every labeled KOI and save
    the resulting view tensors + metadata to disk.

    Returns:
        Metadata DataFrame (one row per successfully processed KOI) that
        is also written to `data/processed/metadata.csv`. Actual tensors
        are saved individually as .npz files (one per KOI) under
        `data/processed/views/`, keyed by kepoi_name, to keep memory
        bounded on a CPU-only dev machine.
    """
    koi_path = Path(cfg.data.external_dir) / "koi_cumulative.csv"
    koi_df = pd.read_csv(koi_path)

    label_map = dict(cfg.data.label_map)
    koi_df = koi_df[koi_df["koi_disposition"].isin(label_map.keys())].copy()
    koi_df["label"] = koi_df["koi_disposition"].map(label_map)

    views_dir = Path(cfg.data.processed_dir) / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    light_curve_dir = Path(cfg.data.raw_dir) / "light_curves"

    records = []
    for _, row in koi_df.iterrows():
        if pd.isna(row["koi_period"]) or pd.isna(row["koi_time0bk"]) or pd.isna(row["koi_duration"]):
            continue  # cannot fold without a full ephemeris

        params = TransitParams(
            period_days=row["koi_period"],
            epoch_bkjd=row["koi_time0bk"],
            duration_hours=row["koi_duration"],
        )
        lc_path = light_curve_dir / f"kic_{int(row['kepid'])}.csv"
        result = process_single_koi(lc_path, params, cfg)
        if result is None:
            continue

        out_path = views_dir / f"{row['kepoi_name']}.npz"
        np.savez_compressed(out_path, **result)

        records.append(
            {
                "kepid": int(row["kepid"]),
                "kepoi_name": row["kepoi_name"],
                "label": int(row["label"]),
                "koi_disposition": row["koi_disposition"],
                "view_path": str(out_path),
            }
        )

    metadata = pd.DataFrame.from_records(records)
    metadata_path = Path(cfg.data.processed_dir) / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    logger.info(
        f"Processed {len(metadata)}/{len(koi_df)} KOIs successfully. "
        f"Label balance: {metadata['label'].value_counts().to_dict()}"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    build_processed_dataset(cfg)


if __name__ == "__main__":
    main()
