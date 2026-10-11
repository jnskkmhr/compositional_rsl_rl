# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-independent config for a vector environment with policy and target observations."""

from dataclasses import dataclass

from compositional_rl.algorithm import CompositionalPPOAlgorithmCfg
from compositional_rl.loss import LossesCfg, SupervisedLossCfg
from compositional_rl.network import ActorHeadCfg, CriticHeadCfg, MLPNetworkCfg, NetworkCfg
from compositional_rl.runner import CompositionalRunnerCfg


@dataclass(kw_only=True)
class EstimatorNetworksCfg(NetworkCfg):
    """Share a supervised estimator between the actor and critic."""

    estimator: MLPNetworkCfg | None = None


@dataclass(kw_only=True)
class EstimatorLossesCfg(LossesCfg):
    """Supervise the estimator using the environment's target observation group."""

    estimate: SupervisedLossCfg | None = None


def make_config() -> CompositionalRunnerCfg:
    """Build a config; pass dataclasses.asdict(cfg) to the upstream OnPolicyRunner.

    The environment must implement the RSL-RL VecEnv protocol and provide a vector
    policy observation and a target observation with three values per environment.
    """
    return CompositionalRunnerCfg(
        networks=EstimatorNetworksCfg(
            actor=ActorHeadCfg(input=["policy", "estimator"], hidden_dims=[256, 128]),
            critic=CriticHeadCfg(input=["policy", "estimator"], hidden_dims=[256, 128]),
            estimator=MLPNetworkCfg(input=["policy"], output_dim=3, hidden_dims=[128, 64]),
        ),
        loss=EstimatorLossesCfg(
            estimate=SupervisedLossCfg(
                trains=["estimator"], prediction="estimator", target="target", optimizer="joint"
            ),
        ),
        algorithm=CompositionalPPOAlgorithmCfg(entropy_coef=0.005),
    )
