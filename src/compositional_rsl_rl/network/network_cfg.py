# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-independent configuration for named RSL-RL network terms."""

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass(kw_only=True)
class GaussianDistributionCfg:
    """Gaussian action distribution with a learned, state-independent scale."""

    class_name: str = "GaussianDistribution"
    init_std: float = 1.0
    std_type: str = "scalar"


@dataclass(kw_only=True)
class HeteroscedasticGaussianDistributionCfg(GaussianDistributionCfg):
    """Gaussian action distribution with a state-dependent scale."""

    class_name: str = "HeteroscedasticGaussianDistribution"


@dataclass(kw_only=True)
class SquashedGaussianDistributionCfg:
    """State-dependent Gaussian followed by a tanh action transform."""

    class_name: str = "SquashedGaussianDistribution"
    log_std_min: float = -10.0
    log_std_max: float = 2.0


@dataclass(kw_only=True)
class CategoricalValueDistributionCfg:
    """Categorical return support, independent of the network architecture."""

    class_name: str = "CategoricalValueDistribution"
    num_bins: int = 101
    value_min: float = -5.0
    value_max: float = 5.0


@dataclass(kw_only=True)
class MLPNetworkCfg:
    """One named term; input names reference observations or other network terms."""

    GaussianDistributionCfg: ClassVar = GaussianDistributionCfg
    HeteroscedasticGaussianDistributionCfg: ClassVar = HeteroscedasticGaussianDistributionCfg
    SquashedGaussianDistributionCfg: ClassVar = SquashedGaussianDistributionCfg
    CategoricalValueDistributionCfg: ClassVar = CategoricalValueDistributionCfg

    input: list[str]
    class_name: str = "MLPModel"
    output_dim: int | tuple[int, ...] | None = None
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "elu"
    obs_normalization: bool = False
    distribution_cfg: (
        GaussianDistributionCfg | SquashedGaussianDistributionCfg | CategoricalValueDistributionCfg | dict | None
    ) = None
    frozen: bool = False
    pretrained_cfg: dict | None = None


@dataclass(kw_only=True)
class ActorHeadCfg(MLPNetworkCfg):
    """Actor term whose output dimension is supplied by the environment."""

    distribution_cfg: GaussianDistributionCfg | SquashedGaussianDistributionCfg | dict | None = field(
        default_factory=GaussianDistributionCfg
    )


@dataclass(kw_only=True)
class CriticHeadCfg(MLPNetworkCfg):
    """Scalar value or Q term; the learner supplies actions for an action-conditioned critic."""


@dataclass(kw_only=True)
class RecurrentActorHeadCfg(ActorHeadCfg):
    """Actor term using the upstream GRU/LSTM model protocol."""

    class_name: str = "RNNModel"
    rnn_type: str = "lstm"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1


@dataclass(kw_only=True)
class RecurrentCriticHeadCfg(CriticHeadCfg):
    """Critic term using the upstream GRU/LSTM model protocol."""

    class_name: str = "RNNModel"
    rnn_type: str = "lstm"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1


@dataclass(kw_only=True)
class CNNNetworkCfg(MLPNetworkCfg):
    """Image-processing term using an upstream CNNModel configuration dictionary."""

    class_name: str = "CNNModel"
    cnn_cfg: dict


@dataclass(kw_only=True)
class TensorNetworkCfg(MLPNetworkCfg):
    """Configure a TensorNetwork subclass by import path, without manager registration."""

    class_name: str = field()
    feature_dim: int = 128
    hidden_dims: list[int] = field(default_factory=lambda: [128])


@dataclass(kw_only=True)
class NetworkCfg:
    """Extend with named network fields; None disables an optional term."""

    actor: ActorHeadCfg
    critic: CriticHeadCfg
