"""Tests for src/models/astronet_lite.py — shape correctness, gradient
flow, and error handling for the dual-branch CNN."""

from __future__ import annotations

import copy

import pytest
import torch

from src.models.astronet_lite import AstroNetLite


@pytest.fixture
def model(small_model_config):
    return AstroNetLite.from_config(small_model_config)


@pytest.fixture
def batch(small_model_config):
    batch_size = 8
    gview = torch.randn(batch_size, 1, small_model_config.data.global_view_bins)
    lview = torch.randn(batch_size, 1, small_model_config.data.local_view_bins)
    return gview, lview, batch_size


class TestForwardPass:
    def test_output_shape(self, model, batch):
        gview, lview, batch_size = batch
        logits = model(gview, lview)
        assert logits.shape == (batch_size,)
        assert logits.dtype == torch.float32

    def test_predict_proba_in_valid_range(self, model, batch):
        gview, lview, batch_size = batch
        probs = model.predict_proba(gview, lview)
        assert probs.shape == (batch_size,)
        assert torch.all((probs >= 0) & (probs <= 1))

    def test_attention_disabled_still_works(self, small_model_config, batch):
        cfg = copy.deepcopy(small_model_config)
        cfg.model.attention = False
        model_no_attn = AstroNetLite.from_config(cfg)
        gview, lview, batch_size = batch
        logits = model_no_attn(gview, lview)
        assert logits.shape == (batch_size,)

    def test_side_features_path(self, small_model_config, batch):
        gview, lview, batch_size = batch
        model_with_side = AstroNetLite.from_config(small_model_config, n_side_features=4)
        side_feats = torch.randn(batch_size, 4)
        logits = model_with_side(gview, lview, side_feats)
        assert logits.shape == (batch_size,)

    def test_missing_side_features_raises(self, small_model_config, batch):
        gview, lview, _ = batch
        model_with_side = AstroNetLite.from_config(small_model_config, n_side_features=4)
        with pytest.raises(ValueError, match="side_features"):
            model_with_side(gview, lview)


class TestGradientFlow:
    def test_all_parameters_receive_gradients(self, model, batch):
        gview, lview, batch_size = batch
        logits = model(gview, lview)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones(batch_size))
        loss.backward()

        params_without_grad = [name for name, p in model.named_parameters() if p.grad is None]
        assert not params_without_grad, f"Parameters with no gradient: {params_without_grad}"


class TestModelSize:
    def test_parameter_count_is_t4_friendly(self, model):
        """Sanity check the model stays well within a size appropriate for
        the stated Colab T4 compute budget (README §6) — this test would
        fail loudly if a future architecture change accidentally made the
        model orders of magnitude larger."""
        n_params = model.count_parameters()
        assert n_params < 5_000_000, (
            f"Model has {n_params:,} parameters — unexpectedly large for the "
            "stated T4 compute budget; verify config.yaml model.* settings."
        )
