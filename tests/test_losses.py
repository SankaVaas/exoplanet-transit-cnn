"""Tests for src/training/losses.py (focal loss)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.training.losses import FocalLoss, build_loss


class TestFocalLoss:
    def test_gamma_zero_equals_alpha_weighted_bce(self):
        torch.manual_seed(0)
        logits = torch.randn(16)
        targets = torch.randint(0, 2, (16,)).float()

        fl = FocalLoss(gamma=0.0, alpha=0.5)
        loss_focal = fl(logits, targets)
        loss_bce_scaled = 0.5 * F.binary_cross_entropy_with_logits(logits, targets)
        assert abs(loss_focal.item() - loss_bce_scaled.item()) < 1e-5

    def test_alpha_one_on_all_positive_targets_equals_full_bce(self):
        torch.manual_seed(0)
        logits = torch.randn(16)
        targets = torch.ones(16)
        fl = FocalLoss(gamma=0.0, alpha=1.0)
        loss_focal = fl(logits, targets)
        loss_bce = F.binary_cross_entropy_with_logits(logits, targets)
        assert abs(loss_focal.item() - loss_bce.item()) < 1e-5

    def test_higher_gamma_downweights_easy_examples(self):
        easy_logits = torch.tensor([5.0, -5.0])
        easy_targets = torch.tensor([1.0, 0.0])
        gamma0_loss = FocalLoss(gamma=0.0, alpha=0.5)(easy_logits, easy_targets)
        gamma2_loss = FocalLoss(gamma=2.0, alpha=0.5)(easy_logits, easy_targets)
        assert gamma2_loss.item() < gamma0_loss.item()

    def test_shape_mismatch_raises(self):
        fl = FocalLoss()
        with pytest.raises(ValueError, match="shape mismatch"):
            fl(torch.randn(4), torch.randn(5))

    @pytest.mark.parametrize("alpha", [-0.1, 1.1])
    def test_invalid_alpha_raises(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            FocalLoss(alpha=alpha)

    def test_negative_gamma_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            FocalLoss(gamma=-1.0)


class TestBuildLoss:
    def test_focal_factory(self):
        loss_fn = build_loss("focal", focal_gamma=2.0, focal_alpha=0.75)
        assert isinstance(loss_fn, FocalLoss)

    def test_bce_factory(self):
        loss_fn = build_loss("bce")
        assert isinstance(loss_fn, torch.nn.BCEWithLogitsLoss)

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            build_loss("not_a_real_loss")
