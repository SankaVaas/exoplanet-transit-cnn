"""
Download the NASA Exoplanet Archive cumulative KOI (Kepler Objects of
Interest) table — this is the label source for the whole project.

Each row is a TCE (Threshold Crossing Event) with a disposition:
CONFIRMED, CANDIDATE, or FALSE POSITIVE, plus the orbital/transit
parameters (period, epoch, duration) needed to phase-fold the
corresponding light curve in src/data/preprocess.py.

Usage:
    python -m src.data.download_koi_catalog --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import load_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# NASA Exoplanet Archive TAP (Table Access Protocol) sync endpoint.
# The cumulative KOI table is queried via ADQL rather than scraping the
# website UI, for a stable, scriptable, citable data source.
NASA_EXOPLANET_ARCHIVE_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Columns we actually need — kepid to join with light curve files, the
# disposition label, and the transit parameters required for phase-folding.
KOI_COLUMNS = [
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "koi_period",       # orbital period, days
    "koi_time0bk",       # transit epoch (BKJD)
    "koi_duration",       # transit duration, hours
    "koi_depth",           # transit depth, ppm
    "koi_prad",              # planet radius, Earth radii (side-feature)
    "koi_srad",                # stellar radius, Solar radii (side-feature)
    "koi_steff",                 # stellar effective temperature, K (side-feature)
    "koi_slogg",                   # stellar surface gravity (side-feature)
]


def build_adql_query(columns: list[str]) -> str:
    """Build the ADQL query string for the cumulative KOI table."""
    col_str = ", ".join(columns)
    return f"select {col_str} from cumulative"


def download_koi_catalog(output_path: Path, columns: list[str] = KOI_COLUMNS) -> pd.DataFrame:
    """Query and save the cumulative KOI table.

    Args:
        output_path: where to write the resulting CSV
            (default: data/external/koi_cumulative.csv).
        columns: which columns to request — kept minimal and documented
            above rather than pulling all ~150 columns of the full table.

    Returns:
        The downloaded table as a DataFrame (also written to disk).

    Raises:
        requests.HTTPError: if the archive query fails (bad column name,
            service outage, etc.) — surfaced explicitly rather than
            silently returning an empty/partial table.
    """
    query = build_adql_query(columns)
    params = {"query": query, "format": "csv"}

    logger.info("Querying NASA Exoplanet Archive cumulative KOI table...")
    response = requests.get(NASA_EXOPLANET_ARCHIVE_TAP_URL, params=params, timeout=60)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response.text, encoding="utf-8")

    df = pd.read_csv(output_path)
    logger.info(f"Downloaded {len(df):,} KOI rows -> {output_path}")

    n_missing_labels = df["koi_disposition"].isna().sum()
    if n_missing_labels > 0:
        logger.warning(f"{n_missing_labels} rows have no koi_disposition label.")

    disposition_counts = df["koi_disposition"].value_counts()
    logger.info(f"Disposition breakdown:\n{disposition_counts.to_string()}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_path = Path(cfg.data.external_dir) / "koi_cumulative.csv"
    download_koi_catalog(output_path)


if __name__ == "__main__":
    main()
