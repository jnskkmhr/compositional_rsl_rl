# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-independent configuration for the upstream RSL-RL runner."""

from dataclasses import dataclass, field

from compositional_rl.algorithm.algorithm_cfg import CompositionalPPOAlgorithmCfg
from compositional_rl.loss.loss_cfg import LossesCfg
from compositional_rl.network.network_cfg import NetworkCfg


@dataclass(kw_only=True)
class CompositionalRunnerCfg:
    """Serialize with dataclasses.asdict before passing to OnPolicyRunner.

    Environment integrations may also use their own dataclass serializer. The
    runtime consumes ordinary dictionaries, independent of any simulator.
    """

    networks: NetworkCfg
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
    experiment_name: str = "compositional_rl"
    run_name: str = ""
    logger: str = "tensorboard"
    neptune_project: str = "compositional_rl"
    wandb_project: str = "compositional_rl"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    torch_compile_mode: str | None = None
