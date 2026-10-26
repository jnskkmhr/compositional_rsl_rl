# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-independent FlashSAC algorithm and head configuration."""

from dataclasses import dataclass

from .algorithm_cfg import SchedulerCfg


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
    symmetry_cfg: dict | None = None
    curriculum_cfg: dict[str, SchedulerCfg | dict] | None = None
