"""
Evaluation metrics with bootstrapped confidence intervals.

Per README §3, this project reports precision/recall/F1/ROC-AUC/PR-AUC/
Brier score with 95% CIs rather than a bare accuracy number — a single
point estimate on a few thousand test examples can look very different on
a re-sample, and reporting that uncertainty honestly is part of what
distinguishes rigorous ML work from a headline accuracy figure.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_point_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute all point-estimate metrics for one set of predictions.

    Args:
        y_true: binary ground-truth labels, shape (N,).
        y_prob: predicted probabilities in [0, 1], shape (N,).
        threshold: decision threshold for precision/recall/F1 (ROC-AUC,
            PR-AUC, and Brier score are threshold-independent).

    Returns:
        dict of metric name -> value.

    Raises:
        ValueError: if y_true/y_prob have mismatched shapes, or y_prob
            contains values outside [0, 1] (a common bug when logits are
            accidentally passed instead of sigmoid outputs).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError(
            "y_prob contains values outside [0, 1] — did you pass raw logits "
            "instead of sigmoid(logits)?"
        )

    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, y_prob),
    }

    # AUC metrics are undefined with only one class present in y_true —
    # return NaN explicitly rather than letting sklearn raise mid-pipeline.
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap a 95% (or other) confidence interval for a single metric.

    Args:
        y_true: ground-truth labels, shape (N,).
        y_prob: predicted probabilities, shape (N,).
        metric_fn: callable(y_true_sample, y_prob_sample) -> float, e.g.
            `lambda yt, yp: average_precision_score(yt, yp)`.
        n_iterations: number of bootstrap resamples (config:
            evaluation.bootstrap_iterations).
        confidence: confidence level for the interval, e.g. 0.95 for a 95% CI.
        seed: RNG seed for reproducibility.

    Returns:
        (point_estimate, ci_lower, ci_upper) computed on the full sample
        for the point estimate, and via percentile bootstrap for the CI.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    point_estimate = metric_fn(y_true, y_prob)

    boot_scores = []
    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        y_true_boot, y_prob_boot = y_true[idx], y_prob[idx]
        if len(np.unique(y_true_boot)) < 2:
            continue  # skip degenerate resamples with only one class
        try:
            boot_scores.append(metric_fn(y_true_boot, y_prob_boot))
        except ValueError:
            continue  # metric undefined for this resample (e.g. no positives)

    if len(boot_scores) < n_iterations * 0.5:
        # If more than half the bootstrap resamples were unusable, the CI
        # would be unreliable — surface this rather than silently reporting
        # a CI computed from too few resamples (relevant on small test sets).
        raise RuntimeError(
            f"Only {len(boot_scores)}/{n_iterations} bootstrap resamples were "
            "usable — test set is likely too small or too imbalanced for a "
            "reliable CI on this metric."
        )

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_scores, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
    return float(point_estimate), ci_lower, ci_upper


def full_evaluation_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Produce the complete metrics report used in reports/results.md:
    point estimate + bootstrapped 95% CI for every metric.

    Returns:
        dict mapping metric name -> {"point": ..., "ci_lower": ..., "ci_upper": ...}
    """
    metric_fns = {
        "precision": lambda yt, yp: precision_score(yt, (yp >= threshold).astype(int), zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, (yp >= threshold).astype(int), zero_division=0),
        "f1": lambda yt, yp: f1_score(yt, (yp >= threshold).astype(int), zero_division=0),
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
        "brier_score": brier_score_loss,
    }

    report = {}
    for name, fn in metric_fns.items():
        point, lo, hi = bootstrap_metric_ci(y_true, y_prob, fn, n_bootstrap, seed=seed)
        report[name] = {"point": point, "ci_lower": lo, "ci_upper": hi}
    return report
