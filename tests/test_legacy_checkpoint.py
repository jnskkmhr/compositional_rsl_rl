# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ordinary PPO checkpoints can resume independent compositional MLP policies."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from test_composition import Environment, config, rollout

from compositional_rsl_rl import CompositionalPPO
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner


def _legacy_configs() -> tuple[dict, dict]:
    """Equivalent normalized MLPs, with reversed graph declaration order."""
    actor = {
        "class_name": "MLPModel",
        "hidden_dims": [8],
        "obs_normalization": True,
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5},
    }
    critic = {"class_name": "MLPModel", "hidden_dims": [8], "obs_normalization": True}
    plain = config()
    plain.pop("networks")
    plain.pop("loss")
    plain.update(actor=actor, critic=critic, obs_groups={"actor": ["policy"], "critic": ["policy", "target"]})
    plain["algorithm"]["class_name"] = "PPO"
    # Ordinary PPO clips each head separately; compositional PPO clips all trainable terms together.
    # Disable clipping to isolate checkpoint/optimizer equivalence in the next-update comparison.
    plain["algorithm"]["max_grad_norm"] = float("inf")
    graph = copy.deepcopy(plain)
    graph.pop("actor")
    graph.pop("critic")
    graph["networks"] = {
        "critic": dict(critic, input=["policy", "target"]),
        "actor": dict(actor, input=["policy"]),
    }
    graph["algorithm"]["class_name"] = "compositional_rsl_rl:CompositionalPPO"
    return plain, graph


def test_legacy_runner_resume_matches_next_ppo_update(tmp_path: Path) -> None:
    """Legacy resume retains outputs, normalization, Adam moments, and the next PPO update."""
    plain, graph = _legacy_configs()
    env = Environment()
    source = OnPolicyRunner(env, plain, log_dir=None, device="cpu")
    source.learn(1)
    path = tmp_path / "legacy.pt"
    source.save(str(path))
    env.common_step_counter = 99
    restored = OnPolicyRunner(env, graph, log_dir=None, device="cpu")
    with pytest.warns(RuntimeWarning, match="estimating common_step_counter"):
        restored.load(str(path))
    assert env.common_step_counter == 4
    assert restored.current_learning_iteration == source.current_learning_iteration
    obs = env.get_observations()
    for name in ("actor", "critic"):
        expected = getattr(source.alg, name)
        actual = getattr(restored.alg, name)
        torch.testing.assert_close(actual(obs), expected(obs), rtol=0, atol=0)
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(actual.head.state_dict()[key], value, rtol=0, atol=0)
        for old, new in zip(expected.parameters(), actual.head.parameters()):
            for key, value in source.alg.optimizer.state[old].items():
                torch.testing.assert_close(restored.alg.optimizer.state[new][key], value, rtol=0, atol=0)
    assert restored.alg.learning_rate == source.alg.learning_rate
    # Collect identical transitions and minibatches, then compare real PPO updates.
    for algo in (source.alg, restored.alg):
        torch.manual_seed(123)
        rollout(algo, Environment())
        algo.update()
    for name in ("actor", "critic"):
        for key, value in getattr(source.alg, name).state_dict().items():
            torch.testing.assert_close(restored.alg.networks.terms[name].state_dict()[key], value)
    # Subsequent native saves retain the exact counter and load through the existing path.
    env.common_step_counter = 12345
    restored.save(str(tmp_path / "native.pt"))
    env.common_step_counter = 0
    restored.load(str(tmp_path / "native.pt"))
    assert env.common_step_counter == 12345


def test_legacy_weights_only_and_exact_counter() -> None:
    """Actor-only transfer leaves critic/optimizer/progress alone; resume honors a stored counter."""
    plain, graph = _legacy_configs()
    env = Environment()
    obs = env.get_observations()
    plain["multi_gpu"] = None
    source = PPO.construct_algorithm(obs, env, plain, "cpu")
    checkpoint = copy.deepcopy(source.save())
    checkpoint.update(iter=100, environment_training_state={"common_step_counter": 123})
    env.common_step_counter = 77
    target = CompositionalPPO.construct_algorithm(obs, env, graph, "cpu")
    critic_before = copy.deepcopy(target.networks.terms["critic"].state_dict())
    assert not target.load(checkpoint, {"actor": True, "iteration": False}, True)
    assert env.common_step_counter == 77
    assert not target.optimizer.state
    for key, value in critic_before.items():
        torch.testing.assert_close(target.networks.terms["critic"].state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(target.actor(obs), source.actor(obs), rtol=0, atol=0)
    assert target.load(checkpoint, {"actor": True, "critic": True, "iteration": True}, True)
    assert env.common_step_counter == 123
    assert not target.optimizer.state
    assert "network_state_dict" not in checkpoint  # Loading never mutates the caller's checkpoint.


def test_legacy_rejects_incompatible_optimizer_and_graph() -> None:
    """Reject unsafe resumes before overwriting policy weights."""
    plain, graph = _legacy_configs()
    env = Environment()
    obs = env.get_observations()
    plain["multi_gpu"] = None
    source = PPO.construct_algorithm(obs, env, plain, "cpu")
    rollout(source, env)
    source.update()
    checkpoint = copy.deepcopy(source.save())
    checkpoint["iter"] = 0
    target = CompositionalPPO.construct_algorithm(obs, env, graph, "cpu")
    before = copy.deepcopy(target.networks.state_dict())
    state = next(iter(checkpoint["optimizer_state_dict"]["state"].values()))
    state["exp_avg"] = torch.zeros(99)
    with pytest.raises(ValueError, match="optimizer state shape"):
        target.load(checkpoint, None, True)
    for key, value in before.items():
        torch.testing.assert_close(target.networks.state_dict()[key], value, rtol=0, atol=0)
    complex_target = CompositionalPPO.construct_algorithm(obs, env, config(), "cpu")
    with pytest.raises(ValueError, match="two independent"):
        complex_target.load(checkpoint, None, True)
