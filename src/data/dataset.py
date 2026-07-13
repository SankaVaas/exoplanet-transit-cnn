"""
PyTorch Dataset + grouped splitting for the preprocessed global/local view
tensors produced by src/data/preprocess.py.

Splitting is grouped by `kepid` (host star), NOT by individual KOI row.
This matters: several KOIs (candidate planets) can share the same host
star, and if two TCEs from the same star land on opposite sides of a
train/test split, the model can implicitly "memorize" that star's stellar
variability signature rather than learning genuine transit-vs-non-transit
features — inflating test performance in a way that would not generalize.
This is exactly the kind of subtle leakage a careless implementation
would miss, and exactly what a careful README should call out.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from src.utils.config import ConfigDict
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TransitViewDataset(Dataset):
    """Loads (global_view, local_view, label) triples from metadata.csv rows."""

    def __init__(self, metadata: pd.DataFrame):
        """
        Args:
            metadata: DataFrame with columns `view_path`, `label` (0/1), and
                `kepid` — a subset of the full metadata.csv produced by
                build_processed_dataset(), typically one of the
                train/val/test splits from `grouped_train_val_test_split`.
        """
        self.metadata = metadata.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[idx]
        npz = np.load(row["view_path"])
        global_view = torch.from_numpy(npz["global_view"]).float().unsqueeze(0)  # (1, L)
        local_view = torch.from_numpy(npz["local_view"]).float().unsqueeze(0)    # (1, L)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return global_view, local_view, label


def grouped_train_val_test_split(
    metadata: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    group_col: str = "kepid",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split metadata into train/val/test with no `group_col` value appearing
    in more than one split (prevents same-star leakage — see module docstring).

    Args:
        metadata: full metadata DataFrame from build_processed_dataset().
        test_size: fraction of *groups* (stars) held out for the test set.
        val_size: fraction of the *remaining* (post-test) groups held out
            for validation.
        group_col: column defining the grouping unit — `kepid` by default.
        seed: RNG seed for the split, sourced from config.yaml `project.seed`.

    Returns:
        (train_df, val_df, test_df) — each a subset of `metadata` with
        disjoint `group_col` values.

    Raises:
        ValueError: if any star appears in more than one split (a sanity
            check that would only fail if this function itself had a bug —
            included because leakage bugs are exactly the kind of mistake
            that silently inflates reported metrics).
    """
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(metadata, groups=metadata[group_col]))
    trainval_df = metadata.iloc[trainval_idx]
    test_df = metadata.iloc[test_idx]

    relative_val_size = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
    train_idx, val_idx = next(gss_val.split(trainval_df, groups=trainval_df[group_col]))
    train_df = trainval_df.iloc[train_idx]
    val_df = trainval_df.iloc[val_idx]

    train_groups = set(train_df[group_col])
    val_groups = set(val_df[group_col])
    test_groups = set(test_df[group_col])
    overlap = (train_groups & val_groups) | (train_groups & test_groups) | (val_groups & test_groups)
    if overlap:
        raise ValueError(
            f"Group leakage detected across splits for {group_col} values: "
            f"{list(overlap)[:5]}... This should never happen with "
            "GroupShuffleSplit — investigate immediately."
        )

    logger.info(
        f"Split sizes -> train: {len(train_df)} ({len(train_groups)} stars), "
        f"val: {len(val_df)} ({len(val_groups)} stars), "
        f"test: {len(test_df)} ({len(test_groups)} stars)"
    )
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        balance = df["label"].value_counts(normalize=True).to_dict()
        logger.info(f"  {name} label balance: {balance}")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_splits(cfg: ConfigDict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience loader: reads metadata.csv and applies the grouped split
    using parameters from config.yaml `data.test_size` / `data.val_size`."""
    metadata_path = Path(cfg.data.processed_dir) / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found. Run `python -m src.data.preprocess` first."
        )
    metadata = pd.read_csv(metadata_path)
    return grouped_train_val_test_split(
        metadata,
        test_size=cfg.data.test_size,
        val_size=cfg.data.val_size,
        group_col=cfg.data.split_by,
        seed=cfg.project.seed,
    )
