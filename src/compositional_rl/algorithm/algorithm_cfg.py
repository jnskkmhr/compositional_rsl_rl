# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-independent configuration for compositional PPO."""

from dataclasses import dataclass


@dataclass(kw_only=True)
class CompositionalPPOAlgorithmCfg:
    """Select compositional PPO through the upstream runner's algorithm resolver."""

    class_name: str = "compositional_rl:CompositionalPPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    learning_rate: float = 1.0e-3
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    optimizer: str = "adam"
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    normalize_advantage_per_mini_batch: bool = False
    use_mixed_precision: bool = False
    rnd_cfg: dict | None = None
    symmetry_cfg: dict | None = None
    grad_reduce_bucket_mb: float = 25
