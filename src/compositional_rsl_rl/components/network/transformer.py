# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Transformer terms for fixed-length token observations, with optional validity masks."""

from typing import Any

import torch
from torch import nn

from compositional_rsl_rl.network import TensorNetwork


class _AttentionBlock(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, feedforward_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.norm_attention = nn.LayerNorm(feature_dim)
        self.qkv = nn.Linear(feature_dim, 3 * feature_dim)
        self.projection = nn.Linear(feature_dim, feature_dim)
        self.norm_feedforward = nn.LayerNorm(feature_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(feature_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, feature_dim),
        )

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, length, width = tokens.shape
        qkv = self.qkv(self.norm_attention(tokens)).reshape(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (query @ key.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
        # Avoid softmax(-inf, ...) on empty clouds, including its undefined backward pass.
        scores = torch.where(valid.any(dim=-1)[:, None, None, None], scores, 0.0)
        attention = torch.softmax(scores, dim=-1).masked_fill(~valid[:, None, None, :], 0.0)
        mixed = (attention @ value).transpose(1, 2).reshape(batch, length, width)
        tokens = tokens + self.projection(mixed)
        tokens = tokens + self.feedforward(self.norm_feedforward(tokens))
        return torch.where(valid.unsqueeze(-1), tokens, 0.0)


class TransformerEncoder(nn.Module):
    """Encode tokens with pre-norm self-attention and masked mean pooling.

    Positional embeddings identify token indices, not simulation timestamps. Dropout
    is omitted so repeated PPO evaluations use the same policy for fixed parameters.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        feature_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 256,
        validity_channel: bool = False,
        use_positional_encoding: bool = True,
    ) -> None:
        """Configure attention; optionally treat the final input channel as validity > 0.5."""
        super().__init__()
        if len(input_shape) != 2 or min(input_shape) < 1:
            raise ValueError("TransformerEncoder expects nonempty [batch, tokens, channels] inputs.")
        if num_heads < 1 or feature_dim < 1 or feature_dim % num_heads:
            raise ValueError("feature_dim must be positive and divisible by num_heads.")
        if num_layers < 1 or feedforward_dim < 1:
            raise ValueError("num_layers and feedforward_dim must be positive.")
        input_dim = input_shape[-1] - int(validity_channel)
        if input_dim < 1:
            raise ValueError("TransformerEncoder requires data channels in addition to validity.")
        self.validity_channel = validity_channel
        self.input_projection = nn.Linear(input_dim, feature_dim)
        positions = torch.zeros(1, input_shape[0], feature_dim)
        if use_positional_encoding:
            self.positions = nn.Parameter(positions)
            nn.init.normal_(self.positions, std=0.02)
        else:
            self.register_buffer("positions", positions)
        self.layers = nn.ModuleList(
            [_AttentionBlock(feature_dim, num_heads, feedforward_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Exclude padded data from attention and pooling, even when it contains NaNs."""
        if self.validity_channel:
            valid = inputs[..., -1] > 0.5
            inputs = inputs[..., :-1]
        else:
            valid = torch.ones_like(inputs[..., 0], dtype=torch.bool)
        inputs = torch.where(valid.unsqueeze(-1), inputs, 0.0)
        tokens = self.input_projection(inputs) + self.positions
        for layer in self.layers:
            tokens = layer(tokens, valid)
        tokens = torch.where(valid.unsqueeze(-1), self.norm(tokens), 0.0)
        return tokens.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)


class TransformerNetwork(TensorNetwork):
    """A Transformer encoder and MLP readout using the standard graph protocol."""

    def __init__(
        self,
        *args: Any,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 256,
        validity_channel: bool = False,
        use_positional_encoding: bool = True,
        **kwargs: Any,
    ) -> None:
        """Configure within-observation attention; there is no persistent temporal state."""
        self.encoder_options = {
            "num_heads": num_heads,
            "num_layers": num_layers,
            "feedforward_dim": feedforward_dim,
            "validity_channel": validity_channel,
            "use_positional_encoding": use_positional_encoding,
        }
        super().__init__(*args, **kwargs)

    def build_encoder(self, input_shape: tuple[int, ...], feature_dim: int, activation: str) -> nn.Module:
        """Build self-attention with GELU feed-forward blocks; activation config controls the readout."""
        return TransformerEncoder(input_shape, feature_dim, **self.encoder_options)
