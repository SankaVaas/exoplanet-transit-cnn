"""
AstroNet-Lite: a dual-branch 1D-CNN for transit classification.

Architecture summary (see README §5 and config.yaml `model.*`):

    global_view (1, 2001) --[Conv1D stack]--> global_features
    local_view  (1, 201)  --[Conv1D stack]--> local_features
                                 |
                        [self-attention over local_features]
                                 |
    [optional stellar side-features (radius, Teff, logg, ...)] --[MLP]--> side_features
                                 |
              concat(global_features, attended_local_features, side_features)
                                 |
                          [fusion MLP head]
                                 |
                          logit (pre-sigmoid)

This is a lighter version of Shallue & Vanderburg's AstroNet, sized to
train on a single Colab T4 (mixed precision) in well under the free-tier
session limit — see README §6 for the stated compute budget. The model
outputs raw logits (not probabilities); use `torch.sigmoid` or the
`predict_proba` convenience method to get calibrated-input probabilities
(actual calibration is a separate temperature-scaling step, see
src/evaluation/calibration.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.config import ConfigDict


class ConvBlock1D(nn.Module):
    """Conv1D -> ReLU -> Conv1D -> ReLU -> MaxPool1D, the repeated unit used
    in both the global and local branches. Two convs per block (rather than
    one) mirrors AstroNet's design, giving more representational capacity
    per downsampling step without a large parameter increase."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBranch(nn.Module):
    """A stack of ConvBlock1D layers, e.g. the global or local view branch."""

    def __init__(self, channels: list[int], kernel_size: int, pool_size: int):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in channels:
            layers.append(ConvBlock1D(in_ch, out_ch, kernel_size, pool_size))
            in_ch = out_ch
        self.layers = nn.ModuleList(layers)
        self.out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the final feature map, shape (B, C, L') — pooling is NOT
        applied here (caller decides global-pool vs. attention pooling)."""
        for layer in self.layers:
            x = layer(x)
        return x


class LocalAttentionPool(nn.Module):
    """Self-attention over the local-view feature map's temporal axis,
    followed by attention-weighted pooling to a fixed-size vector.

    Motivation (see README §2): transit ingress/egress shape asymmetry is a
    known discriminator between genuine planetary transits and eclipsing
    binaries / grazing geometries. A plain global-average-pool over the
    local branch treats every time step equally; attention lets the model
    learn to weight the ingress/egress edges (or center, or asymmetric
    wings) differently per example, rather than assuming a fixed
    hand-designed pooling window.
    """

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(channels)
        # Learned query vector — a single "summary token" that attends over
        # the sequence, similar in spirit to a CLS token pooling scheme.
        self.query = nn.Parameter(torch.randn(1, 1, channels) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: feature map of shape (B, C, L') from ConvBranch.

        Returns:
            Pooled feature vector of shape (B, C).
        """
        x = x.transpose(1, 2)  # (B, L', C) — MultiheadAttention expects (B, seq, embed)
        x = self.norm(x)
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)  # (B, 1, C)
        attended, _ = self.attn(query, x, x)  # (B, 1, C)
        return attended.squeeze(1)  # (B, C)


class AstroNetLite(nn.Module):
    """Dual-branch CNN + attention for transit classification.

    See module docstring for the architecture diagram. Instantiate via
    `AstroNetLite.from_config(cfg)` to stay in sync with config.yaml rather
    than hand-passing every hyperparameter.
    """

    def __init__(
        self,
        global_view_len: int,
        local_view_len: int,
        global_conv_channels: list[int],
        local_conv_channels: list[int],
        kernel_size: int = 5,
        pool_size: int = 5,
        dropout: float = 0.3,
        use_attention: bool = True,
        attention_heads: int = 4,
        fc_hidden: list[int] | None = None,
        n_side_features: int = 0,
    ):
        super().__init__()
        fc_hidden = fc_hidden or [128, 64]

        self.global_branch = ConvBranch(global_conv_channels, kernel_size, pool_size)
        self.local_branch = ConvBranch(local_conv_channels, kernel_size, pool_size)
        self.use_attention = use_attention

        if use_attention:
            self.local_pool = LocalAttentionPool(
                self.local_branch.out_channels, num_heads=attention_heads, dropout=dropout
            )
        else:
            self.local_pool = None  # falls back to adaptive average pool in forward()

        self.global_gap = nn.AdaptiveAvgPool1d(1)

        self.n_side_features = n_side_features
        if n_side_features > 0:
            self.side_mlp = nn.Sequential(
                nn.Linear(n_side_features, 16),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            side_out_dim = 16
        else:
            self.side_mlp = None
            side_out_dim = 0

        fusion_in_dim = self.global_branch.out_channels + self.local_branch.out_channels + side_out_dim

        fc_layers = []
        in_dim = fusion_in_dim
        for hidden_dim in fc_hidden:
            fc_layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_dim = hidden_dim
        fc_layers.append(nn.Linear(in_dim, 1))  # single logit, binary classification
        self.fusion_head = nn.Sequential(*fc_layers)

    @classmethod
    def from_config(cls, cfg: ConfigDict, n_side_features: int = 0) -> "AstroNetLite":
        """Build the model directly from a loaded config.yaml, keeping the
        architecture and the config file as a single source of truth."""
        m = cfg.model
        return cls(
            global_view_len=cfg.data.global_view_bins,
            local_view_len=cfg.data.local_view_bins,
            global_conv_channels=list(m.global_conv_channels),
            local_conv_channels=list(m.local_conv_channels),
            kernel_size=m.kernel_size,
            pool_size=m.pool_size,
            dropout=m.dropout,
            use_attention=m.attention,
            attention_heads=m.attention_heads,
            fc_hidden=list(m.fc_hidden),
            n_side_features=n_side_features,
        )

    def forward(
        self,
        global_view: torch.Tensor,
        local_view: torch.Tensor,
        side_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            global_view: (B, 1, global_view_len)
            local_view: (B, 1, local_view_len)
            side_features: optional (B, n_side_features) stellar parameters.

        Returns:
            logits: (B,) — raw scores, apply sigmoid for probabilities.
        """
        g = self.global_branch(global_view)
        g = self.global_gap(g).squeeze(-1)  # (B, C_global)

        l = self.local_branch(local_view)
        if self.use_attention:
            l = self.local_pool(l)  # (B, C_local)
        else:
            l = nn.functional.adaptive_avg_pool1d(l, 1).squeeze(-1)

        features = [g, l]
        if self.side_mlp is not None:
            if side_features is None:
                raise ValueError(
                    "Model was built with n_side_features > 0 but no side_features "
                    "tensor was passed to forward()."
                )
            features.append(self.side_mlp(side_features))

        fused = torch.cat(features, dim=1)
        logits = self.fusion_head(fused).squeeze(-1)  # (B,)
        return logits

    def predict_proba(self, *args, **kwargs) -> torch.Tensor:
        """Convenience wrapper: forward() + sigmoid, for inference-time use.
        NOTE: these are raw sigmoid outputs, not temperature-calibrated —
        see src/evaluation/calibration.py for calibrated probabilities used
        in reports/results.md."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(*args, **kwargs))

    def count_parameters(self) -> int:
        """Total trainable parameter count — reported in the README/training
        logs so reviewers can sanity-check the model size against the
        stated Colab T4 compute budget."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
