"""
Grad-CAM-style saliency for the local-view branch (1D adaptation).

Standard Grad-CAM (Selvaraju et al., 2017) is defined for 2D conv feature
maps; here it's adapted to the 1D-conv local branch of AstroNetLite so we
can visualize *which part of the transit window* (ingress, bottom, egress,
or the surrounding out-of-transit baseline) drove a given prediction —
directly supporting the README's point about ingress/egress asymmetry
being physically meaningful, and giving reviewers a way to sanity-check
that the model is actually looking at the transit rather than an
unrelated artifact in the window.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.astronet_lite import AstroNetLite


class LocalViewGradCAM:
    """Computes 1D Grad-CAM saliency over the local-view branch's final
    conv layer for a given input example.

    Usage:
        cam = LocalViewGradCAM(model)
        saliency = cam.compute(global_view, local_view)  # shape (local_view_len,)
    """

    def __init__(self, model: AstroNetLite):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Hook the last ConvBlock1D in the local branch to capture both
        its forward activations and the gradient flowing back into them."""
        target_layer = self.model.local_branch.layers[-1]

        def forward_hook(module, inp, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def compute(
        self, global_view: torch.Tensor, local_view: torch.Tensor
    ) -> np.ndarray:
        """Compute the Grad-CAM saliency map for a single example.

        Args:
            global_view: shape (1, 1, global_view_len) — a single example
                (batch dimension of 1).
            local_view: shape (1, 1, local_view_len) — same example.

        Returns:
            Saliency map upsampled to `local_view_len`, normalized to
            [0, 1], where higher values indicate regions of the local view
            that most influenced the model's prediction.

        Raises:
            ValueError: if inputs don't have batch size 1 — Grad-CAM here
                is defined per-example, not batched, to keep the hook
                logic unambiguous (a batched version would need per-example
                gradient isolation, which is easy to get silently wrong).
        """
        if global_view.size(0) != 1 or local_view.size(0) != 1:
            raise ValueError(
                f"LocalViewGradCAM.compute expects batch size 1, got "
                f"global_view {global_view.shape}, local_view {local_view.shape}"
            )

        self.model.eval()
        self.model.zero_grad()

        logit = self.model(global_view, local_view)
        logit.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Hooks did not fire — check that the target layer is on the "
                "forward path actually executed (e.g. attention enabled/disabled "
                "doesn't skip the conv branch itself, only its pooling)."
            )

        # Grad-CAM weight per channel = global-average-pooled gradient.
        weights = self.gradients.mean(dim=2, keepdim=True)  # (1, C, 1)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)  # (L',)
        cam = torch.relu(cam)  # only positive influence, per Grad-CAM convention

        cam_np = cam.numpy()
        target_len = local_view.shape[-1]
        upsampled = np.interp(
            np.linspace(0, 1, target_len),
            np.linspace(0, 1, len(cam_np)),
            cam_np,
        )

        cam_min, cam_max = upsampled.min(), upsampled.max()
        if cam_max - cam_min < 1e-8:
            return np.zeros(target_len, dtype=np.float32)  # flat CAM — no discriminative region
        return ((upsampled - cam_min) / (cam_max - cam_min)).astype(np.float32)
