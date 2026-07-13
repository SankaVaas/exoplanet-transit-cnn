"""
Reproducibility utilities.

Deterministic training is a stated goal of this project (see README §3 on
honest reporting) — results should be re-derivable from a fixed seed, not
just "roughly reproducible."
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Seed python, numpy, and torch (if installed) RNGs.

    Args:
        seed: seed value, sourced from config.yaml `project.seed`.
        deterministic_torch: if True, also configures torch/cuDNN for
            deterministic algorithms. This can slow down training slightly
            on GPU (cuDNN autotune is disabled) but is worth it for a
            portfolio project where reproducibility is part of the pitch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch not installed yet (e.g. during initial env setup) — skip.
        pass


def get_device(preference: str = "auto") -> "torch.device":  # noqa: F821
    """Resolve the training device.

    Args:
        preference: "auto" (default, prefers CUDA if available), "cpu",
            or "cuda". Matches `training.device` in config.yaml.

    Returns:
        torch.device — will be a CUDA device on Colab T4 when available,
        else CPU, so the same code path runs unmodified in both
        environments as described in the README's compute-budget section.
    """
    import torch

    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("training.device='cuda' requested but no CUDA device found.")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
