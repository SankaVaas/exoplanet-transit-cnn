"""Tests for src/data/dataset.py — most importantly, the grouped-split
leakage prevention described in the module's docstring and README §5."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import grouped_train_val_test_split


@pytest.fixture
def synthetic_metadata():
    """Synthetic metadata with MULTIPLE KOIs sharing the same kepid (star),
    which is exactly the situation that could leak across splits if
    splitting were done per-row instead of per-group."""
    rng = np.random.default_rng(0)
    records = []
    for star_idx in range(100):
        kepid = 1000 + star_idx
        n_kois_for_star = rng.integers(1, 4)  # some stars have multiple KOIs
        for koi_idx in range(n_kois_for_star):
            records.append(
                {
                    "kepid": kepid,
                    "kepoi_name": f"K{star_idx:05d}.{koi_idx:02d}",
                    "label": int(rng.random() < 0.3),
                    "view_path": f"/fake/path/K{star_idx:05d}.{koi_idx:02d}.npz",
                }
            )
    return pd.DataFrame.from_records(records)


class TestGroupedSplit:
    def test_no_star_appears_in_multiple_splits(self, synthetic_metadata):
        train_df, val_df, test_df = grouped_train_val_test_split(
            synthetic_metadata, test_size=0.15, val_size=0.15, seed=42
        )
        train_stars = set(train_df["kepid"])
        val_stars = set(val_df["kepid"])
        test_stars = set(test_df["kepid"])

        assert train_stars.isdisjoint(val_stars)
        assert train_stars.isdisjoint(test_stars)
        assert val_stars.isdisjoint(test_stars)

    def test_split_sizes_roughly_match_requested_fractions(self, synthetic_metadata):
        train_df, val_df, test_df = grouped_train_val_test_split(
            synthetic_metadata, test_size=0.15, val_size=0.15, seed=42
        )
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(synthetic_metadata)
        # Rough tolerance since group sizes vary (some stars have 1-3 KOIs)
        assert 0.5 < len(train_df) / total < 0.85
        assert 0.05 < len(test_df) / total < 0.30

    def test_all_rows_accounted_for(self, synthetic_metadata):
        train_df, val_df, test_df = grouped_train_val_test_split(synthetic_metadata, seed=42)
        all_kepoi = set(train_df["kepoi_name"]) | set(val_df["kepoi_name"]) | set(test_df["kepoi_name"])
        assert all_kepoi == set(synthetic_metadata["kepoi_name"])

    def test_reproducible_with_same_seed(self, synthetic_metadata):
        split1 = grouped_train_val_test_split(synthetic_metadata, seed=42)
        split2 = grouped_train_val_test_split(synthetic_metadata, seed=42)
        for df1, df2 in zip(split1, split2):
            assert list(df1["kepoi_name"]) == list(df2["kepoi_name"])
