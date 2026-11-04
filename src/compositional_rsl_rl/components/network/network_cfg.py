# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configs for reusable structured-input networks; no simulator imports or registries."""

from dataclasses import dataclass, field

from compositional_rsl_rl.network import MLPNetworkCfg, TensorNetworkCfg


@dataclass(kw_only=True)
class PointNetNetworkCfg(TensorNetworkCfg):
    """Point-cloud input [batch, points, 4], containing XYZ and a validity flag."""

    class_name: str = "compositional_rsl_rl.components.network:PointNetNetwork"
    pointnet_hidden_dims: list[int] = field(default_factory=lambda: [64, 128])


@dataclass(kw_only=True)
class TransformerNetworkCfg(TensorNetworkCfg):
    """Fixed-length tokens [batch, tokens, channels], optionally ending in validity."""

    class_name: str = "compositional_rsl_rl.components.network:TransformerNetwork"
    num_heads: int = 4
    num_layers: int = 2
    feedforward_dim: int = 256
    validity_channel: bool = False
    use_positional_encoding: bool = True


@dataclass(kw_only=True)
class AttentionMapNetworkCfg(MLPNetworkCfg):
    """XYZ-map cross-attention queried by the other input vectors.

    No readout by default: output_dim must equal feature_dim, or twice feature_dim
    when attach_global=True. Set hidden_dims to add an MLP readout of arbitrary width.
    """

    class_name: str = "compositional_rsl_rl.components.network:AttentionMapNetwork"
    map_input: str = "elevation_map"
    map_shape: tuple[int, int, int] | None = None
    feature_dim: int = 64
    num_heads: int = 16
    cnn_channels: int = 16
    cnn_downsample: bool = True
    cnn_normalization: str = "batch"
    attach_global: bool = False
    global_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    hidden_dims: list[int] = field(default_factory=list)


@dataclass(kw_only=True)
class ResidualNetworkCfg(MLPNetworkCfg):
    """A named residual feature term; output_dim sets each member's feature width."""

    class_name: str = "compositional_rsl_rl.components.network:ResidualNetwork"
    num_blocks: int = 2
    num_ensembles: int = 1
    hidden_dims: list[int] = field(default_factory=list)
    activation: str = "relu"
