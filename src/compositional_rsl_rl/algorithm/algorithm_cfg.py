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


@dataclass(kw_only=True)
class CompositionalFastSACAlgorithmCfg:
    """Holosoma FastSAC with separate termination flags and bounded replay."""

    class_name: str = "compositional_rsl_rl:CompositionalFastSAC"
    critic_action_input: str = "critic"
    gamma: float = 0.97
    target_tau: float = 0.125
    actor_update_period: int = 4
    learning_rate: float = 3e-4
    num_bins: int = 101
    value_max: float = 20.0
    temperature_initial: float = 0.001
    target_entropy_ratio: float = 0.0
    use_autotune: bool = True
    weight_decay: float = 0.001
    max_grad_norm: float = 0.0
    obs_normalization: bool = True
    action_scale: float | list[float] = 1.0
    use_amp: bool = False
    amp_dtype: str = "bf16"
    grad_reduce_bucket_mb: float = 25
    symmetry_cfg: dict | None = None
    curriculum_cfg: dict[str, SchedulerCfg | dict] | None = None


@dataclass(kw_only=True)
class CompositionalSuperMimicAlgorithmCfg:
    """Native model-based learning with task-owned differentiable reward and state coordinates."""

    task_cfg: dict
    class_name: str = "compositional_rsl_rl.algorithm:CompositionalSuperMimic"
    replay_buffer_size: int = 1_000_000
    max_replay_memory_fraction: float | None = None
    updates_per_step: float = 0.4
    num_world_model_updates: int = 10
    num_policy_updates: int = 1
    world_model_window: int = 8
    world_model_batch_size: int = 2048
    horizon: int = 32
    policy_batch_size: int = 4096
    batch_size_is_global: bool = True
    learning_rate: float = 2e-4
    lr_decay_iterations: int = 250_000
    lr_final_ratio: float = 0.03
    gamma: float = 0.99
    gae_lambda: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    warmup_steps: int = 64
    world_model_warmup_iterations: int = 500
    action_noise: float = 0.1
    max_step_reward: float | None = None
    use_mixed_precision: bool = False
    max_grad_norm: float = 1.0
