# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configs for reusable structured-input networks; no simulator imports or registries."""

from dataclasses import dataclass, field

from compositional_rl.network import TensorNetworkCfg


@dataclass(kw_only=True)
class PointNetNetworkCfg(TensorNetworkCfg):
    """Point-cloud input [batch, points, 4], containing XYZ and a validity flag."""

    class_name: str = "compositional_rl.mdp.network:PointNetNetwork"
    pointnet_hidden_dims: list[int] = field(default_factory=lambda: [64, 128])


@dataclass(kw_only=True)
class TransformerNetworkCfg(TensorNetworkCfg):
    """Fixed-length tokens [batch, tokens, channels], optionally ending in validity."""

    class_name: str = "compositional_rl.mdp.network:TransformerNetwork"
    num_heads: int = 4
    num_layers: int = 2
    feedforward_dim: int = 256
    validity_channel: bool = False
    use_positional_encoding: bool = True
