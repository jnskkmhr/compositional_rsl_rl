# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PointNet terms adapted from IsaacLab's g1_29dof_perception/agents/pointnet.py.

The shared XYZ MLP and validity-masked max pooling retain the source encoder's
semantics. The simulator-independent TensorNetwork supplies the RSL-RL adapter.
"""

import torch
from torch import nn
from typing import Any

from compositional_rl.network import TensorNetwork
from rsl_rl.modules import MLP


class PointNetEncoder(nn.Module):
    """Encode [batch, points, 4] XYZ/validity inputs with permutation-invariant pooling."""

    def __init__(self, hidden_dims: tuple[int, ...] | list[int], feature_dim: int, activation: str) -> None:
        """Construct the shared point MLP; the fourth input channel is a validity flag."""
        super().__init__()
        self.point_mlp = MLP(3, feature_dim, hidden_dims, activation)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Ignore invalid XYZ, including NaN padding; empty clouds produce zero features."""
        valid = points[..., 3:4] > 0.5
        xyz = torch.where(valid, points[..., :3], 0.0)
        features = self.point_mlp(xyz).masked_fill(~valid, -torch.inf)
        pooled = features.max(dim=1).values
        return torch.where(valid.any(dim=1), pooled, 0.0)


class PointNetNetwork(TensorNetwork):
    """A named point-cloud encoder and MLP readout for compositional graphs."""

    def __init__(
        self, *args: Any, pointnet_hidden_dims: tuple[int, ...] | list[int] = (64, 128), **kwargs: Any
    ) -> None:
        """Configure the shared point MLP, then construct the standard tensor term."""
        self.pointnet_hidden_dims = pointnet_hidden_dims
        super().__init__(*args, **kwargs)

    def build_encoder(self, input_shape: tuple[int, ...], feature_dim: int, activation: str) -> nn.Module:
        """Build the source-compatible encoder after validating its XYZ/validity layout."""
        if len(input_shape) != 2 or input_shape[-1] != 4:
            raise ValueError(f"PointNetNetwork expects [batch, points, 4], got input shape {input_shape}.")
        return PointNetEncoder(self.pointnet_hidden_dims, feature_dim, activation)
