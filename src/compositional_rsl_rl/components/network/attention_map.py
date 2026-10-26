# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Attention-based map encoding with explicit map and robot-state observation inputs.

Implements CNN map tokens, state-query cross-attention, and optional global context
using standard PyTorch modules. See docs/attention_map.md for the reference method,
input coordinate conventions, and differences from its combined actor/critic model.
"""

from __future__ import annotations

import copy
import math

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from tensordict import TensorDict
from torch import nn


class _CrossAttention(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.output = nn.Linear(feature_dim, feature_dim)

    def forward(self, query: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, width = tokens.shape
        q = self.query(query).reshape(batch, self.num_heads, 1, self.head_dim)
        k = self.key(tokens).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(tokens).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        weights = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        features = (weights @ v).reshape(batch, width)
        return self.output(features), weights.squeeze(2)


class AttentionMapEncoder(nn.Module):
    """Encode an XYZ map using robot-state queries and single-query multi-head attention."""

    def __init__(
        self,
        query_dim: int,
        feature_dim: int = 64,
        num_heads: int = 16,
        cnn_channels: int = 16,
        cnn_downsample: bool = True,
        cnn_normalization: str = "batch",
        attach_global: bool = False,
        global_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        activation: str = "elu",
    ) -> None:
        """Build XYZ convolutions and state-query attention; no temporal state is maintained."""
        super().__init__()
        if query_dim < 1 or feature_dim < 1 or num_heads < 1 or feature_dim % num_heads:
            raise ValueError("query_dim must be positive and feature_dim must be positive and divisible by num_heads.")
        if cnn_channels < 1:
            raise ValueError("cnn_channels must be positive.")
        if cnn_normalization not in ("batch", "none"):
            raise ValueError("cnn_normalization must be 'batch' or 'none'.")
        if attach_global and not global_hidden_dims:
            raise ValueError("The global-context MLP requires at least one hidden layer.")
        self.attach_global = attach_global
        self.output_dim = feature_dim * (2 if attach_global else 1)
        self.map_cnn = nn.Sequential(
            nn.Conv2d(
                3,
                cnn_channels,
                kernel_size=5,
                padding=2,
                stride=2 if cnn_downsample else 1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(cnn_channels) if cnn_normalization == "batch" else nn.Identity(),
            nn.Conv2d(
                cnn_channels,
                feature_dim,
                kernel_size=3 if cnn_downsample else 5,
                padding=1 if cnn_downsample else 2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(feature_dim) if cnn_normalization == "batch" else nn.Identity(),
        )
        self.state_projection = nn.Linear(query_dim, feature_dim)
        self.global_encoder = (
            MLP(feature_dim, feature_dim, global_hidden_dims, activation) if attach_global else nn.Identity()
        )
        self.query_projection = nn.Linear(2 * feature_dim, feature_dim) if attach_global else nn.Identity()
        self.attention = _CrossAttention(feature_dim, num_heads)

    def encode(self, xyz_map: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return map features and weights [batch, heads, encoded_height, encoded_width]."""
        spatial_features = self.map_cnn(xyz_map.permute(0, 3, 1, 2))
        tokens = spatial_features.flatten(start_dim=2).transpose(1, 2)
        query = self.state_projection(state)
        if self.attach_global:
            global_features = self.global_encoder(tokens).max(dim=1).values
            query = self.query_projection(torch.cat([global_features, query], dim=-1))
            attended, weights = self.attention(query, tokens)
            features = torch.cat([global_features, attended], dim=-1)
        else:
            features, weights = self.attention(query, tokens)
        return features, weights.reshape(
            weights.shape[0],
            weights.shape[1],
            spatial_features.shape[2],
            spatial_features.shape[3],
        )

    def forward(self, xyz_map: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Return only features so policy outputs retain the standard runner contract."""
        return self.encode(xyz_map, state)[0]


class AttentionMapNetwork(MLPModel):
    """A feed-forward AME graph term with one XYZ map and one or more vector query inputs.

    With hidden_dims=[], output_dim must match the encoder width: feature_dim, or
    2 * feature_dim with global context. Nonempty hidden_dims adds an MLP readout.
    Empirical normalization, when enabled, applies only to vector query inputs.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        map_input: str = "elevation_map",
        map_shape: tuple[int, int, int] | list[int] | None = None,
        feature_dim: int = 64,
        num_heads: int = 16,
        cnn_channels: int = 16,
        cnn_downsample: bool = True,
        cnn_normalization: str = "batch",
        attach_global: bool = False,
        global_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        hidden_dims: tuple[int, ...] | list[int] = (),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Infer map/query dimensions and construct an encoder compatible with RSL-RL models."""
        nn.Module.__init__(self)
        self.input_groups = list(obs_groups[obs_set])
        if len(set(self.input_groups)) != len(self.input_groups):
            raise ValueError("AttentionMapNetwork inputs must be unique.")
        if map_input not in self.input_groups or len(self.input_groups) < 2:
            raise ValueError("AttentionMapNetwork requires map_input and at least one vector query input.")
        if distribution_cfg is not None:
            raise ValueError("AttentionMapNetwork is an encoder; put the action distribution on an actor head.")
        self.map_input = map_input
        shape = tuple(obs[map_input].shape[len(obs.batch_size) :])
        self.map_shape = tuple(map_shape) if map_shape is not None else shape
        if len(self.map_shape) != 3 or self.map_shape[-1] != 3 or any(size < 1 for size in self.map_shape):
            raise ValueError("Expected map_shape=(height, width, 3); flattened maps require an explicit map_shape.")
        if shape not in (self.map_shape, (math.prod(self.map_shape),)):
            raise ValueError(f"Map input shape {shape} does not match XYZ map_shape {self.map_shape}.")
        self.obs_groups = [name for name in self.input_groups if name != map_input]
        for name in self.obs_groups:
            if len(obs[name].shape) != len(obs.batch_size) + 1 or obs[name].shape[-1] < 1:
                raise ValueError(f"Query input '{name}' must be a nonempty vector per observation.")
        self.obs_dim = sum(obs[name].shape[-1] for name in self.obs_groups)
        self.obs_normalization = obs_normalization
        self.obs_normalizer = EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()
        self.distribution = None
        self.encoder = AttentionMapEncoder(
            self.obs_dim,
            feature_dim,
            num_heads,
            cnn_channels,
            cnn_downsample,
            cnn_normalization,
            attach_global,
            global_hidden_dims,
            activation,
        )
        if hidden_dims:
            self.mlp = MLP(self.encoder.output_dim, output_dim, hidden_dims, activation)
        else:
            if output_dim != self.encoder.output_dim:
                raise ValueError(
                    f"Without a readout, output_dim must be {self.encoder.output_dim}; "
                    "set hidden_dims to project to a different output dimension."
                )
            self.mlp = nn.Identity()
        self.input_sizes = [math.prod(obs[name].shape[len(obs.batch_size) :]) for name in self.input_groups]

    def _inputs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        xyz_map = obs[self.map_input].reshape(-1, *self.map_shape)
        state = torch.cat([obs[name] for name in self.obs_groups], dim=-1).reshape(-1, self.obs_dim)
        return xyz_map, self.obs_normalizer(state)

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Encode each observation independently, retaining rollout batch dimensions."""
        xyz_map, state = self._inputs(obs)
        return self.encoder(xyz_map, state).reshape(*obs.batch_size, self.encoder.output_dim)

    @torch.no_grad()
    def attention_weights(self, obs: TensorDict) -> torch.Tensor:
        """Inspect per-head map weights in eval mode, without changing policy outputs or buffers."""
        if self.training:
            raise RuntimeError("Call eval() before requesting diagnostic attention weights.")
        xyz_map, state = self._inputs(obs)
        weights = self.encoder.encode(xyz_map, state)[1]
        return weights.reshape(*obs.batch_size, *weights.shape[1:])

    def as_jit(self) -> nn.Module:
        """Export the complete multi-input term under the graph's flattened input contract."""
        return _AttentionMapExport(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Export deterministic features and standard RSL-RL input metadata."""
        return _OnnxAttentionMapExport(self, verbose)


class _AttentionMapExport(nn.Module):
    __constants__ = ("query_starts", "query_ends")

    def __init__(self, model: AttentionMapNetwork) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(model.encoder)
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.map_shape = list(model.map_shape)
        query_starts = []
        query_ends = []
        offset = 0
        for name, size in zip(model.input_groups, model.input_sizes):
            if name == model.map_input:
                self.map_start = offset
                self.map_end = offset + size
            else:
                query_starts.append(offset)
                query_ends.append(offset + size)
            offset += size
        self.query_starts = tuple(query_starts)
        self.query_ends = tuple(query_ends)
        self.input_size = offset
        self.eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        state = torch.cat([obs[:, a:b] for a, b in zip(self.query_starts, self.query_ends)], dim=-1)
        shape = [obs.shape[0]]
        shape.extend(self.map_shape)
        xyz_map = obs[:, self.map_start : self.map_end].reshape(shape)
        return self.mlp(self.encoder(xyz_map, self.obs_normalizer(state)))

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxAttentionMapExport(_AttentionMapExport):
    is_recurrent = False

    def __init__(self, model: AttentionMapNetwork, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["features"]
