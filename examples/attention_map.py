# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-independent AME config: policy vectors, critic vectors, and an XYZ elevation map."""

from dataclasses import dataclass

from compositional_rsl_rl.algorithm import CompositionalPPOAlgorithmCfg
from compositional_rsl_rl.mdp.network import AttentionMapNetworkCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg, NetworkCfg
from compositional_rsl_rl.runner import CompositionalRunnerCfg


@dataclass(kw_only=True)
class MyNetworksCfg(NetworkCfg):
    """One map encoder shared by actor and critic, queried only by actor-visible observations."""

    map_encoder: AttentionMapNetworkCfg | None = None


def make_config() -> CompositionalRunnerCfg:
    """Build AME with no auxiliary loss; both PPO policy and value objectives train the encoder."""
    return CompositionalRunnerCfg(
        networks=MyNetworksCfg(
            actor=ActorHeadCfg(input=["policy", "map_encoder"], hidden_dims=[512, 256, 128]),
            critic=CriticHeadCfg(input=["critic", "map_encoder"], hidden_dims=[512, 256, 128]),
            map_encoder=AttentionMapNetworkCfg(
                input=["elevation_map", "policy"],
                map_input="elevation_map",
                map_shape=(33, 21, 3),
                output_dim=64,
                feature_dim=64,
                num_heads=16,
                cnn_channels=16,
                cnn_downsample=True,
                cnn_normalization="batch",
                attach_global=False,
                hidden_dims=[],
                obs_normalization=False,
            ),
        ),
        algorithm=CompositionalPPOAlgorithmCfg(entropy_coef=0.005),
    )
