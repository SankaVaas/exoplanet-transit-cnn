"""
Consistent logging setup across all pipeline stages (download, preprocess,
train, evaluate). Avoids each script re-inventing print-statement logging,
which is a common source of "worked on my machine, silent on Colab" bugs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: str | Path | None = None) -> logging.Logger:
    """Create (or fetch) a logger with console + optional file output.

    Args:
        name: logger name, conventionally __name__ of the calling module.
        log_dir: if given, also writes a `<name>.log` file into this
            directory (created if missing). Typically `outputs/logs`.

    Returns:
        Configured logging.Logger. Safe to call repeatedly for the same
        name — handlers are not duplicated on re-invocation.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured (e.g. re-imported in a notebook) — don't
        # duplicate handlers, which would duplicate every log line.
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_name = name.replace(".", "_")
        file_handler = logging.FileHandler(log_dir / f"{safe_name}.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
