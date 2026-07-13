"""
Focal loss for binary classification under class imbalance.

Confirmed/candidate transits are a minority class among all vetted TCEs
(see README §3). Plain BCE lets the abundant, easy "false positive" class
dominate the gradient; focal loss (Lin et al., 2017, originally for object
detection) down-weights well-classified examples so the model keeps
learning from the harder, rarer true-transit cases throughout training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss operating on raw logits (numerically stable, uses
    `binary_cross_entropy_with_logits` internally rather than sigmoid + BCE
    separately).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: focusing parameter — higher values down-weight easy examples
            more aggressively. gamma=0 reduces to standard weighted BCE.
        alpha: weight for the positive (transit) class; (1 - alpha) is
            implicitly applied to the negative class. Set > 0.5 to upweight
            the rare positive class, matching config.yaml `training.focal_alpha`.
        reduction: "mean", "sum", or "none".
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, reduction: str = "mean"):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: raw model outputs, shape (B,).
            targets: binary labels (0.0 or 1.0), shape (B,), same dtype/device.

        Returns:
            Scalar loss (if reduction != "none") or per-example loss (B,).
        """
        if logits.shape != targets.shape:
            raise ValueError(f"logits/targets shape mismatch: {logits.shape} vs {targets.shape}")

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_term = (1 - p_t).clamp(min=1e-8) ** self.gamma

        loss = alpha_t * focal_term * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(loss_name: str, focal_gamma: float = 2.0, focal_alpha: float = 0.75) -> nn.Module:
    """Factory matching config.yaml `training.loss`.

    Args:
        loss_name: "focal" or "bce".
        focal_gamma: only used if loss_name == "focal".
        focal_alpha: only used if loss_name == "focal".

    Returns:
        An nn.Module callable as loss(logits, targets).

    Raises:
        ValueError: on an unrecognized loss_name — fails loudly rather than
            silently defaulting to some other loss.
    """
    if loss_name == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown loss '{loss_name}'. Use 'focal' or 'bce'.")
