"""
Temperature scaling calibration (Guo et al., 2017).

Neural networks trained with focal/cross-entropy loss are frequently
overconfident — a model can report 0.95 probability while being correct
only 80% of the time at that confidence level. Temperature scaling learns
a single scalar T > 0 post-hoc (on the validation set, never the test set)
that divides the logits before the sigmoid, correcting this without
touching the model's ranking of predictions (AUC-type metrics are
unaffected; only the probability *values* are recalibrated).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    """Learns a single temperature parameter T to calibrate model logits.

    Usage:
        scaler = TemperatureScaler()
        scaler.fit(val_logits, val_targets)
        calibrated_probs = scaler.calibrate(test_logits)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
        outer_steps: int = 50,
        tol: float = 1e-7,
    ) -> float:
        """Fit temperature by minimizing NLL on held-out (validation) logits.

        Args:
            logits: raw model logits on the validation set, shape (N,).
            targets: binary labels, shape (N,).
            lr: learning rate for the LBFGS optimizer.
            max_iter: max LBFGS iterations *per outer step*.
            outer_steps: number of outer `optimizer.step(closure)` calls.
                IMPORTANT: a single LBFGS `step()` call, even with
                `max_iter` set, frequently terminates early because its
                internal line search satisfies its own per-call tolerance
                before reaching the true optimum — this is a well-known
                PyTorch LBFGS gotcha. Looping several outer steps (the
                pattern used in the original temperature-scaling paper's
                reference implementation) is required for real convergence;
                verified empirically in tests/test_calibration.py against
                a grid-search ground truth.
            tol: stop early if the loss improves by less than this between
                outer steps.

        Returns:
            The fitted temperature value (T > 1 means the model was
            overconfident and predictions are being softened; T < 1 means
            the model was underconfident).

        Raises:
            ValueError: if logits/targets are empty or shapes mismatch —
                fitting calibration on the wrong tensor is a silent-failure
                risk worth guarding against explicitly.
        """
        if logits.shape != targets.shape:
            raise ValueError(f"Shape mismatch: logits {logits.shape} vs targets {targets.shape}")
        if len(logits) == 0:
            raise ValueError("Cannot fit calibration on empty logits.")

        logits = logits.detach()
        targets = targets.detach().float()
        criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), targets)
            loss.backward()
            return loss

        prev_loss = float("inf")
        for _ in range(outer_steps):
            loss = optimizer.step(closure)
            current_loss = float(loss.item())
            if abs(prev_loss - current_loss) < tol:
                break
            prev_loss = current_loss

        # Temperature must stay positive — LBFGS is unconstrained, so clamp
        # defensively after fitting rather than letting a pathological fit
        # produce a negative or zero temperature that would invert probabilities.
        with torch.no_grad():
            self.temperature.clamp_(min=0.05)

        return float(self.temperature.item())

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the fitted temperature and return calibrated probabilities."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(logits))


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) — the standard scalar summary
    of calibration quality, reported alongside the reliability diagram in
    reports/results.md.

    Args:
        y_true: binary labels, shape (N,).
        y_prob: predicted probabilities, shape (N,).
        n_bins: number of equal-width confidence bins.

    Returns:
        ECE value in [0, 1] — lower is better calibrated. A perfectly
        calibrated model has ECE = 0.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_prob, bin_edges[1:-1])

    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        bin_confidence = y_prob[mask].mean()
        bin_accuracy = y_true[mask].mean()
        bin_weight = mask.sum() / n
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)
