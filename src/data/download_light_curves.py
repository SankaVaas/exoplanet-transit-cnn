"""
Download Kepler light curves for every star (kepid) present in the KOI
catalog, via `lightkurve` (which queries MAST under the hood).

This is the slowest and most storage-sensitive step in the pipeline, so it
is deliberately resumable (skips kepids already downloaded) and rate-aware
(MAST throttles aggressive query rates).

Usage:
    python -m src.data.download_light_curves --config config.yaml --limit 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_lightkurve():
    """Lazy import so `--help` and unit tests don't require lightkurve
    (a heavy dependency with its own network calls at import time in some
    versions) to be installed."""
    import lightkurve as lk

    return lk


def download_light_curves_for_star(
    kepid: int,
    output_dir: Path,
    mission: str = "Kepler",
    author: str = "Kepler",
    max_retries: int = 3,
) -> Path | None:
    """Download and cache all available quarters of light curve data for one star.

    Args:
        kepid: Kepler Input Catalog ID (the `kepid` column from the KOI table).
        output_dir: directory to save the stitched, cleaned light curve
            (as a .fits or .csv — see implementation).
        mission: "Kepler" or "TESS", matches `data.mission` in config.yaml.
        author: pipeline author filter passed to lightkurve's search, keeps
            us on the official Kepler-pipeline-processed light curves
            rather than mixing in community-contributed reductions.
        max_retries: network calls to MAST occasionally time out; retry
            with backoff rather than failing the whole batch on one star.

    Returns:
        Path to the saved light curve file, or None if no data was found
        for this kepid (logged as a warning, not a fatal error — some KOIs
        genuinely have no public light curve, e.g. proprietary-period data).
    """
    lk = _get_lightkurve()
    output_path = output_dir / f"kic_{kepid}.csv"
    if output_path.exists():
        return output_path  # resumable — skip already-downloaded stars

    for attempt in range(1, max_retries + 1):
        try:
            search_result = lk.search_lightcurve(
                f"KIC {kepid}", mission=mission, author=author
            )
            if len(search_result) == 0:
                logger.warning(f"No light curves found for KIC {kepid}.")
                return None

            # On the first attempt, use the local astropy/lightkurve cache if
            # present (faster, avoids re-downloading valid files). On any
            # retry, force cache=False: a common real-world failure mode is
            # a PREVIOUS interrupted download leaving a truncated FITS file
            # in the cache, which astropy will happily keep trying to read
            # (and fail identically) on every subsequent attempt unless
            # explicitly told to bypass the cache and re-fetch from MAST.
            use_cache = attempt == 1
            lc_collection = search_result.download_all(cache=use_cache)
            stitched = lc_collection.stitch()

            # Remove NaNs and clear outliers flagged by the mission pipeline's
            # own quality bitmask before we ever touch this data — garbage
            # cadences should not reach the detrending step.
            stitched = stitched.remove_nans().remove_outliers(sigma=6)

            output_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(
                {
                    "time": stitched.time.value,
                    "flux": stitched.flux.value,
                    "flux_err": stitched.flux_err.value,
                }
            )
            df.to_csv(output_path, index=False)
            return output_path

        except Exception as exc:  # noqa: BLE001 — deliberately broad: MAST
            # network errors surface as various exception types across
            # lightkurve/astroquery versions; we want uniform retry behavior.
            wait = 2**attempt
            logger.warning(
                f"KIC {kepid}: attempt {attempt}/{max_retries} failed ({exc}). "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    logger.error(f"KIC {kepid}: all {max_retries} download attempts failed. Skipping.")
    return None


def download_batch(
    kepids: list[int],
    output_dir: Path,
    mission: str = "Kepler",
    author: str = "Kepler",
) -> dict[str, int]:
    """Download light curves for a list of kepids, logging a running summary.

    Returns:
        dict with counts of "succeeded", "failed", "skipped_existing" —
        used both for a human-readable log and as a return value tests can
        assert against.
    """
    stats = {"succeeded": 0, "failed": 0, "skipped_existing": 0}
    for i, kepid in enumerate(kepids, start=1):
        existed_before = (output_dir / f"kic_{kepid}.csv").exists()
        result = download_light_curves_for_star(kepid, output_dir, mission, author)
        if result is None:
            stats["failed"] += 1
        elif existed_before:
            stats["skipped_existing"] += 1
        else:
            stats["succeeded"] += 1

        if i % 25 == 0 or i == len(kepids):
            logger.info(
                f"Progress: {i}/{len(kepids)} | "
                f"succeeded={stats['succeeded']} "
                f"skipped={stats['skipped_existing']} "
                f"failed={stats['failed']}"
            )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of stars to download — use a small "
        "value (e.g. 50) for a fast CPU-side smoke test before committing "
        "to the full multi-hour download.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    koi_path = Path(cfg.data.external_dir) / "koi_cumulative.csv"
    if not koi_path.exists():
        raise FileNotFoundError(
            f"{koi_path} not found. Run `python -m src.data.download_koi_catalog` first."
        )

    koi_df = pd.read_csv(koi_path)
    kepids = sorted(koi_df["kepid"].dropna().unique().astype(int).tolist())
    if args.limit:
        kepids = kepids[: args.limit]
        logger.info(f"--limit set: downloading first {len(kepids)} stars only.")

    output_dir = Path(cfg.data.raw_dir) / "light_curves"
    mission_name = "Kepler" if cfg.data.mission == "kepler" else "TESS"

    logger.info(f"Downloading light curves for {len(kepids)} stars ({mission_name})...")
    stats = download_batch(kepids, output_dir, mission=mission_name, author=mission_name)
    logger.info(f"Done. Final stats: {stats}")


if __name__ == "__main__":
    main()
