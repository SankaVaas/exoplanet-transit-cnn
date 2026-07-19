"""
Download Kepler light curves for every star (kepid) present in the KOI
catalog.

IMPORTANT IMPLEMENTATION NOTE: this module uses `lightkurve` only for its
search step (metadata query — confirmed reliable). Actual file transfer
uses plain `requests` directly against MAST's download API, and FITS
parsing uses `astropy.io.fits` directly, rather than lightkurve's
`download_all()` / `LightCurve` reading.

This was a deliberate fix, not a style choice: during development,
`lightkurve.SearchResult.download_all()` reliably failed/truncated on a
real Windows machine, on every attempt, regardless of cache settings or
astropy's remote_timeout config — while a plain `requests.get()` against
the exact same MAST URI succeeded immediately (466,560/466,560 bytes in
5.2s). That points to a bug/incompatibility inside astroquery's internal
HTTP handling on that environment, not a network, firewall, or MAST
problem. Using `requests` directly for the actual transfer sidesteps it
entirely, and is what this module now does.

Usage:
    python -m src.data.download_light_curves --config config.yaml --limit 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

MAST_DOWNLOAD_ENDPOINT = "https://mast.stsci.edu/api/v0.1/Download/file"


def _get_lightkurve():
    """Lazy import — only used for the search/metadata step, not downloads."""
    import lightkurve as lk

    return lk


def search_products(kepid: int, mission: str = "Kepler", author: str = "Kepler", cadence: str = "long"):
    """Query MAST for available light curve products for one star.

    Args:
        kepid: Kepler Input Catalog ID.
        mission: "Kepler" or "TESS".
        author: pipeline author filter (keeps us on official Kepler-pipeline
            products rather than community reductions).
        cadence: "long" (30-min, default — matches this project's view-
            folding design) or "short" (1-min, ~30x more data, avoid unless
            you specifically need it).

    Returns:
        The lightkurve SearchResult (may be empty — check len() before use).
    """
    lk = _get_lightkurve()
    return lk.search_lightcurve(f"KIC {kepid}", mission=mission, author=author, cadence=cadence)


def download_fits_via_requests(data_uri: str, dest_path: Path, timeout: int = 120) -> None:
    """Download one FITS file directly via `requests`, verifying the
    transferred size against the server's Content-Length before accepting
    the file — this is the specific check that would have caught every
    truncated download that lightkurve's own downloader silently accepted.

    Args:
        data_uri: the `dataURI` value from a lightkurve search result row
            (e.g. "mast:KEPLER/url/missions/kepler/lightcurves/...").
        dest_path: local path to write the FITS file to.
        timeout: per-request timeout in seconds.

    Raises:
        requests.exceptions.RequestException: on HTTP-level failure.
        IOError: if the downloaded byte count doesn't match the server's
            declared Content-Length — a real truncation, caught explicitly
            rather than silently written to disk as if complete.
    """
    url = f"{MAST_DOWNLOAD_ENDPOINT}?uri={data_uri}"
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        expected_size = int(resp.headers.get("Content-Length", -1))

        total = 0
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                total += len(chunk)

    if expected_size > 0 and total != expected_size:
        tmp_path.unlink(missing_ok=True)
        raise IOError(
            f"Truncated download for {dest_path.name}: got {total} bytes, "
            f"expected {expected_size} bytes."
        )

    tmp_path.replace(dest_path)


def read_kepler_light_curve_fits(path: Path) -> dict[str, np.ndarray]:
    """Parse a single-quarter Kepler light curve FITS file directly with
    astropy.io.fits, bypassing lightkurve's own reader.

    Extracts the standard Kepler pipeline columns: TIME, PDCSAP_FLUX
    (Presearch Data Conditioning flux — instrumental-systematics-corrected,
    the standard column used for transit search, matching Shallue &
    Vanderburg 2018's choice), PDCSAP_FLUX_ERR, and SAP_QUALITY (used to
    drop flagged bad cadences).

    Returns:
        dict with "time", "flux", "flux_err" arrays, filtered to
        SAP_QUALITY == 0 (no quality flags raised) and finite values only.
    """
    from astropy.io import fits

    with fits.open(path) as hdul:
        data = hdul["LIGHTCURVE"].data
        time = np.asarray(data["TIME"], dtype=np.float64)
        flux = np.asarray(data["PDCSAP_FLUX"], dtype=np.float64)
        flux_err = np.asarray(data["PDCSAP_FLUX_ERR"], dtype=np.float64)
        quality = np.asarray(data["SAP_QUALITY"], dtype=np.int64)

    mask = (quality == 0) & np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    return {"time": time[mask], "flux": flux[mask], "flux_err": flux_err[mask]}


def download_light_curves_for_star(
    kepid: int,
    output_dir: Path,
    fits_cache_dir: Path,
    mission: str = "Kepler",
    author: str = "Kepler",
    cadence: str = "long",
    max_retries: int = 3,
) -> Path | None:
    """Download all quarters for one star, stitch them, and save as CSV.

    Args:
        kepid: Kepler Input Catalog ID.
        output_dir: directory for the final stitched CSV (time, flux, flux_err).
        fits_cache_dir: directory to cache raw per-quarter FITS files.
        mission, author, cadence: passed to `search_products`.
        max_retries: per-file retry count on transient HTTP failures.

    Returns:
        Path to the saved CSV, or None if no products were found or every
        quarter failed to download/parse (logged, not raised — a single
        bad star should not abort the whole batch).
    """
    output_path = output_dir / f"kic_{kepid}.csv"
    if output_path.exists():
        return output_path  # resumable

    search_result = search_products(kepid, mission, author, cadence)
    if len(search_result) == 0:
        logger.warning(f"No light curves found for KIC {kepid}.")
        return None

    table = search_result.table
    uri_col = "dataURI" if "dataURI" in table.colnames else None
    if uri_col is None:
        logger.error(f"KIC {kepid}: search result has no 'dataURI' column — cannot download.")
        return None

    star_cache_dir = fits_cache_dir / f"kic_{kepid}"
    quarter_arrays = []

    for i in range(len(table)):
        data_uri = str(table[uri_col][i])
        filename = data_uri.rsplit("/", 1)[-1]
        fits_path = star_cache_dir / filename

        if not fits_path.exists():
            for attempt in range(1, max_retries + 1):
                try:
                    download_fits_via_requests(data_uri, fits_path)
                    break
                except (requests.exceptions.RequestException, IOError) as exc:
                    wait = 2**attempt
                    logger.warning(
                        f"KIC {kepid} quarter {i}: attempt {attempt}/{max_retries} "
                        f"failed ({exc}). Retrying in {wait}s..."
                    )
                    time.sleep(wait)
            else:
                logger.error(f"KIC {kepid} quarter {i}: all {max_retries} attempts failed. Skipping this quarter.")
                continue

        try:
            quarter_data = read_kepler_light_curve_fits(fits_path)
        except (OSError, KeyError) as exc:
            logger.warning(f"KIC {kepid} quarter {i}: failed to parse FITS ({exc}). Skipping this quarter.")
            continue

        if len(quarter_data["time"]) == 0:
            continue

        # Normalize each quarter by its own median flux before stitching —
        # different quarters have different absolute flux levels (different
        # CCD position/aperture each quarter), so without this the stitched
        # curve would show large spurious discontinuities at quarter
        # boundaries. This matches lightkurve's own `.stitch()` default
        # behavior (normalize=True).
        median_flux = np.median(quarter_data["flux"])
        if median_flux != 0:
            quarter_data["flux"] = quarter_data["flux"] / median_flux
            quarter_data["flux_err"] = quarter_data["flux_err"] / median_flux

        quarter_arrays.append(quarter_data)

    if not quarter_arrays:
        logger.warning(f"KIC {kepid}: no quarters successfully downloaded/parsed.")
        return None

    combined = {
        key: np.concatenate([q[key] for q in quarter_arrays])
        for key in ["time", "flux", "flux_err"]
    }
    order = np.argsort(combined["time"])
    combined = {k: v[order] for k, v in combined.items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(combined).to_csv(output_path, index=False)
    return output_path


def download_batch(
    kepids: list[int],
    output_dir: Path,
    fits_cache_dir: Path,
    mission: str = "Kepler",
    author: str = "Kepler",
    cadence: str = "long",
) -> dict[str, int]:
    """Download light curves for a list of kepids, with a live progress bar."""
    stats = {"succeeded": 0, "failed": 0, "skipped_existing": 0}

    progress = tqdm(kepids, desc="Downloading light curves", unit="star", dynamic_ncols=True)
    for kepid in progress:
        existed_before = (output_dir / f"kic_{kepid}.csv").exists()
        star_start = time.time()
        result = download_light_curves_for_star(kepid, output_dir, fits_cache_dir, mission, author, cadence)
        star_elapsed = time.time() - star_start

        if result is None:
            stats["failed"] += 1
        elif existed_before:
            stats["skipped_existing"] += 1
        else:
            stats["succeeded"] += 1

        progress.set_postfix(
            ok=stats["succeeded"], skip=stats["skipped_existing"], fail=stats["failed"],
            last=f"{star_elapsed:.1f}s",
        )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cadence", type=str, default="long", choices=["long", "short"])
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
    fits_cache_dir = Path(cfg.data.raw_dir) / "fits_cache"
    mission_name = "Kepler" if cfg.data.mission == "kepler" else "TESS"

    logger.info(f"Downloading light curves for {len(kepids)} stars ({mission_name}, {args.cadence} cadence)...")
    stats = download_batch(kepids, output_dir, fits_cache_dir, mission=mission_name, author=mission_name, cadence=args.cadence)
    logger.info(f"Done. Final stats: {stats}")


if __name__ == "__main__":
    main()