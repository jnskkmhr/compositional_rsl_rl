# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-independent algorithm and learning-parameter schedule configurations."""

from dataclasses import dataclass


@dataclass(kw_only=True)
class SchedulerCfg:
    """Numeric schedule: linear ramp or discrete milestones, measured in completed collection steps."""

    type: str
    params: dict


@dataclass(kw_only=True)
class CompositionalPPOAlgorithmCfg:
    """Select compositional PPO through the upstream runner's algorithm resolver."""

    class_name: str = "compositional_rsl_rl:CompositionalPPO"
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


@dataclass(kw_only=True)
class CompositionalFlashSACAlgorithmCfg:
    """Select the compositional algorithm through the native off-policy runner."""

    class_name: str = "compositional_rsl_rl:CompositionalFlashSAC"
    critic_action_input: str = "critic"
    """Network term receiving the internally supplied action before its computation."""

    gamma: float = 0.99
    target_tau: float = 0.01
    actor_update_period: int = 2
    learning_rate: float = 3e-4
    final_learning_rate: float = 1.5e-4
    decay_updates: int = 100_000
    num_bins: int = 101
    value_max: float = 5.0
    temperature_initial: float = 0.01
    temperature_target_sigma: float = 0.15
    normalize_reward: bool = True
    noise_zeta_exponent: float = 2.0
    noise_max_repeat: int = 16
    action_support: float = 1.0
    use_amp: bool = False
    grad_reduce_bucket_mb: float = 25
    symmetry_cfg: dict | None = None
    curriculum_cfg: dict[str, SchedulerCfg | dict] | None = None
