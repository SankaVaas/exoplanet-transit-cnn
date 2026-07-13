"""
Config loading utility.

Provides a single `load_config()` entry point so every script (download,
preprocess, train, evaluate) reads from the same config.yaml, avoiding
parameter drift between pipeline stages.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigDict(dict):
    """Dict subclass allowing attribute-style access, e.g. cfg.training.epochs.

    Falls back to standard __getitem__ semantics for anything not present,
    raising a clear KeyError rather than a silent None (a common source of
    quietly-wrong ML configs).
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(
                f"Config has no key '{name}'. Available keys: {list(self.keys())}"
            ) from exc
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _wrap(obj: Any) -> Any:
    """Recursively convert nested dicts into ConfigDict for attribute access."""
    if isinstance(obj, dict):
        return ConfigDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def load_config(path: str | Path = "config.yaml") -> ConfigDict:
    """Load config.yaml into a ConfigDict.

    Args:
        path: Path to the YAML config file. Defaults to the project-root
            config.yaml (assumes the script is invoked with the project
            root as the working directory, e.g. `python -m src.training.train`).

    Returns:
        ConfigDict supporting both cfg["training"]["epochs"] and
        cfg.training.epochs access patterns.

    Raises:
        FileNotFoundError: if the config file does not exist at `path`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{path.resolve()}'. "
            "Run scripts from the project root, or pass --config explicitly."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _wrap(raw)


def override_config(cfg: ConfigDict, overrides: dict[str, Any]) -> ConfigDict:
    """Return a copy of cfg with dotted-key overrides applied.

    Example:
        cfg = override_config(cfg, {"training.epochs": 5, "training.batch_size": 16})

    Useful for quick CPU smoke tests without editing config.yaml, e.g. in
    tests/ or a `--smoke-test` CLI flag.
    """
    cfg = copy.deepcopy(cfg)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return cfg
