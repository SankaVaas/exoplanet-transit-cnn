"""Tests for src/evaluation/saliency.py (1D Grad-CAM)."""

from __future__ import annotations

import copy

import pytest
import torch

from src.evaluation.saliency import LocalViewGradCAM
from src.models.astronet_lite import AstroNetLite


@pytest.fixture
def single_example(small_model_config):
    gview = torch.randn(1, 1, small_model_config.data.global_view_bins)
    lview = torch.randn(1, 1, small_model_config.data.local_view_bins)
    return gview, lview


class TestLocalViewGradCAM:
    def test_output_shape_and_normalization(self, small_model_config, single_example):
        model = AstroNetLite.from_config(small_model_config)
        cam_tool = LocalViewGradCAM(model)
        gview, lview = single_example

        saliency = cam_tool.compute(gview, lview)
        assert saliency.shape == (small_model_config.data.local_view_bins,)
        assert saliency.min() >= -1e-6
        assert saliency.max() <= 1.0 + 1e-6

    def test_batch_size_greater_than_one_raises(self, small_model_config):
        model = AstroNetLite.from_config(small_model_config)
        cam_tool = LocalViewGradCAM(model)
        gview = torch.randn(2, 1, small_model_config.data.global_view_bins)
        lview = torch.randn(2, 1, small_model_config.data.local_view_bins)
        with pytest.raises(ValueError, match="batch size 1"):
            cam_tool.compute(gview, lview)

    def test_works_with_attention_disabled(self, small_model_config, single_example):
        cfg = copy.deepcopy(small_model_config)
        cfg.model.attention = False
        model = AstroNetLite.from_config(cfg)
        cam_tool = LocalViewGradCAM(model)
        gview, lview = single_example

        saliency = cam_tool.compute(gview, lview)
        assert saliency.shape == (small_model_config.data.local_view_bins,)
