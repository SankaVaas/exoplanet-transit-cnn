"""
Decision threshold selection.

A fixed 0.5 threshold is a default, not a law of nature — under class
imbalance (rare true transits among many false positives, see README §3),
a well-ranking model (high ROC-AUC/PR-AUC) can still have ALL its
calibrated probabilities fall on one side of 0.5, producing misleading
precision/recall/F1 of exactly 0 despite genuinely useful ranking ability.
This module finds a threshold from the *validation* set (never test) that
optimizes a chosen criterion, to be reported alongside — not instead of —
the threshold-independent metrics.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve


def find_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, criterion: str = "f1"
) -> tuple[float, float]:
    """Find the probability threshold that optimizes a given criterion.

    Args:
        y_true: binary labels on a VALIDATION set (never fit this on test
            data — doing so would leak test information into the
            threshold choice, inflating reported test performance).
        y_prob: predicted probabilities, shape (N,).
        criterion: "f1" (maximize F1 score) or "youden" (maximize
            sensitivity + specificity - 1, a.k.a. Youden's J statistic).

    Returns:
        (best_threshold, best_score).

    Raises:
        ValueError: if y_true contains only one class (threshold selection
            is undefined without both classes present), or criterion is
            unrecognized.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Cannot select a threshold with only one class present in y_true.")

    if criterion == "f1":
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        # precision_recall_curve returns len(thresholds) = len(precisions) - 1
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-12)
        f1_scores = f1_scores[:-1]  # drop the last point (no corresponding threshold)
        if len(thresholds) == 0:
            return 0.5, 0.0
        best_idx = int(np.argmax(f1_scores))
        return float(thresholds[best_idx]), float(f1_scores[best_idx])

    if criterion == "youden":
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        return float(thresholds[best_idx]), float(j_scores[best_idx])

    raise ValueError(f"Unknown criterion '{criterion}'. Use 'f1' or 'youden'.")


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute precision/recall/F1 at an arbitrary threshold — a thin
    convenience wrapper used to report "metrics at 0.5" alongside "metrics
    at the validation-optimal threshold" side by side in reports/results.md,
    so the difference is visible rather than silently substituted."""
    from sklearn.metrics import precision_score, recall_score

    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
