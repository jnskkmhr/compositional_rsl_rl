# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-independent configuration for the upstream RSL-RL runner."""

from dataclasses import dataclass, field

from compositional_rsl_rl.algorithm.algorithm_cfg import CompositionalPPOAlgorithmCfg
from compositional_rsl_rl.algorithm.flash_sac_cfg import (
    CompositionalFlashSACAlgorithmCfg,
)
from compositional_rsl_rl.env_interface import EnvInterfaceCfg
from compositional_rsl_rl.loss.loss_cfg import LossesCfg
from compositional_rsl_rl.network.network_cfg import NetworkCfg


@dataclass(kw_only=True)
class CompositionalRunnerCfg:
    """Serialize with dataclasses.asdict before passing to OnPolicyRunner.

    Environment integrations may also use their own dataclass serializer. The
    runtime consumes ordinary dictionaries, independent of any simulator.
    """

    networks: NetworkCfg
    env_interface: EnvInterfaceCfg | None = None
    loss: LossesCfg = field(default_factory=LossesCfg)
    algorithm: CompositionalPPOAlgorithmCfg = field(default_factory=CompositionalPPOAlgorithmCfg)
    class_name: str = "OnPolicyRunner"
    actor: None = None
    critic: None = None
    obs_groups: dict[str, list[str]] = field(default_factory=dict)
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 30_000
    init_at_random_ep_len: bool = True
    clip_actions: float | None = None
    check_for_nan: bool = True
    save_interval: int = 500
    experiment_name: str = "compositional_rsl_rl"
    run_name: str = ""
    logger: str = "tensorboard"
    neptune_project: str = "compositional_rsl_rl"
    wandb_project: str = "compositional_rsl_rl"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    torch_compile_mode: str | None = None


@dataclass(kw_only=True)
class CompositionalOffPolicyRunnerCfg:
    """Configure native FlashSAC collection and replay independently of PPO rollouts."""

    networks: NetworkCfg
    algorithm: CompositionalFlashSACAlgorithmCfg = field(default_factory=CompositionalFlashSACAlgorithmCfg)
    loss: LossesCfg = field(default_factory=LossesCfg)
    class_name: str = "OffPolicyRunner"
    obs_groups: dict[str, list[str]] = field(default_factory=dict)
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 1
    updates_per_step: int = 4
    batch_size: int = 2048
    replay_capacity: int = 1_000_000
    replay_max_bytes: int = 3072 * 1024**2
    warmup_transitions: int = 100_000
    n_step: int = 3
    max_iterations: int = 100_000
    init_at_random_ep_len: bool = False
    clip_actions: None = None
    check_for_nan: bool = True
    save_interval: int = 1000
    experiment_name: str = "compositional_flash_sac"
    run_name: str = ""
    logger: str = "tensorboard"
    wandb_project: str = "compositional_flash_sac"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    torch_compile_mode: None = None
