# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PointNet/Transformer composition using simulator-independent dataclasses.

The RSL-RL environment provides policy [batch, features], points [batch, points, 4]
(XYZ plus validity), and perception_target [batch, 3] containing task-defined labels.
"""

from dataclasses import dataclass

from compositional_rsl_rl.loss import LossesCfg, SupervisedLossCfg
from compositional_rsl_rl.components.network import PointNetNetworkCfg, TransformerNetworkCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg, NetworkCfg, TensorNetworkCfg
from compositional_rsl_rl.runner import CompositionalRunnerCfg


@dataclass(kw_only=True)
class PerceptionNetworksCfg(NetworkCfg):
    """An encoder shared by both actor and critic."""

    perception: TensorNetworkCfg | None = None


@dataclass(kw_only=True)
class PerceptionLossesCfg(LossesCfg):
    """Optional supervision of the perception term."""

    perception_loss: SupervisedLossCfg | None = None


def make_config(encoder: str = "pointnet") -> CompositionalRunnerCfg:
    """Choose an encoder; serialize the result with dataclasses.asdict for OnPolicyRunner."""
    if encoder == "pointnet":
        perception = PointNetNetworkCfg(input=["points"], output_dim=3, feature_dim=128)
    elif encoder == "transformer":
        perception = TransformerNetworkCfg(
            input=["points"],
            output_dim=3,
            feature_dim=128,
            validity_channel=True,
            use_positional_encoding=False,
        )
    else:
        raise ValueError(f"Unknown encoder: {encoder}")
    return CompositionalRunnerCfg(
        networks=PerceptionNetworksCfg(
            actor=ActorHeadCfg(input=["policy", "perception"], hidden_dims=[256, 128]),
            critic=CriticHeadCfg(input=["policy", "perception"], hidden_dims=[256, 128]),
            perception=perception,
        ),
        loss=PerceptionLossesCfg(
            perception_loss=SupervisedLossCfg(
                trains=["perception"],
                prediction="perception",
                target="perception_target",
                optimizer="joint",
                loss_fn="mse",
            ),
        ),
    )
