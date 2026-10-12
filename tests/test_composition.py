# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral contracts for the standalone package and unchanged RSL-RL runner."""

from __future__ import annotations

import copy
import torch
from pathlib import Path
from tensordict import TensorDict

import onnx
import onnxruntime
import pytest
from compositional_rsl_rl.loss import LossManager

from compositional_rsl_rl import CompositionalPPO, NetworkManager
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import unpad_trajectories


def config(recurrent: str | None = None, separate: bool = False) -> dict:
    """Declare heads before their shared estimator to exercise dependency sorting."""
    head = {"class_name": "MLPModel", "input": ["policy", "estimator"], "hidden_dims": [8]}
    if recurrent:
        head.update(class_name="RNNModel", rnn_type=recurrent, rnn_hidden_dim=8, rnn_num_layers=1)
    return {
        "class_name": "OnPolicyRunner",
        "num_steps_per_env": 4,
        "save_interval": 1,
        "check_for_nan": True,
        "logger": "tensorboard",
        "networks": {
            "actor": dict(head, distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5}),
            "critic": dict(head),
            "estimator": {
                "class_name": "MLPModel",
                "input": ["policy"],
                "output_dim": 3,
                "hidden_dims": [8],
                "obs_normalization": True,
            },
            "decoder": {"class_name": "MLPModel", "input": ["estimator"], "output_dim": 6, "hidden_dims": [8]},
            "disabled": None,
        },
        "loss": {
            "estimate": {
                "trains": ["estimator"],
                "prediction": "estimator",
                "target": "target",
                "optimizer": "separate" if separate else "joint",
                "learning_rate": 0.003 if separate else None,
                "weight": 0.2,
            }
        },
        "algorithm": {
            "class_name": "compositional_rsl_rl:CompositionalPPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 2,
            "schedule": "fixed",
        },
    }


class Environment:
    """Small deterministic-protocol environment, requiring no simulator."""

    def __init__(self) -> None:
        """Initialize the vector environment metadata used by upstream runners."""
        self.num_envs = 4
        self.num_actions = 2
        self.device = "cpu"
        self.max_episode_length = 4
        self.episode_length_buf = torch.zeros(4, dtype=torch.long)
        self.cfg = {}
        self.unwrapped = self
        self.step_dt = 0.02

    def get_observations(self) -> TensorDict:
        """Return policy inputs and supervision targets as separate groups."""
        return TensorDict({"policy": torch.randn(4, 6), "target": torch.ones(4, 3)}, batch_size=[4])

    def step(self, actions: torch.Tensor) -> tuple:
        """Advance staggered episode boundaries."""
        self.episode_length_buf += 1
        dones = (self.episode_length_buf + torch.arange(4)) % 4 == 0
        return self.get_observations(), torch.randn(4), dones, {}


def make_algorithm(recurrent: str | None = None, separate: bool = False) -> tuple[CompositionalPPO, Environment]:
    """Construct through the entrypoint used by the unchanged runner."""
    env = Environment()
    return CompositionalPPO.construct_algorithm(env.get_observations(), env, config(recurrent, separate), "cpu"), env


def rollout(algo: CompositionalPPO, env: Environment) -> None:
    """Collect real upstream transitions including staggered episode boundaries."""
    with torch.inference_mode():
        obs = env.get_observations()
        for _ in range(4):
            actions = algo.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            algo.process_env_step(obs, rewards, dones, extras)
        algo.compute_returns(obs)


@pytest.mark.parametrize("recurrent", [None, "gru", "lstm"])
@pytest.mark.parametrize("separate", [False, True])
def test_update_and_optimizer_ownership(recurrent: str | None, separate: bool) -> None:
    """Shared estimators update exactly once under the selected owner, including recurrent PPO."""
    algo, env = make_algorithm(recurrent, separate)
    assert algo.actor.terms["estimator"] is algo.critic.terms["estimator"]
    optimizers = [algo.optimizer, *algo.losses.optimizers.values()]
    parameters = [p for opt in optimizers for group in opt.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    estimator_before = copy.deepcopy(algo.networks.terms["estimator"].state_dict())
    decoder_before = copy.deepcopy(algo.networks.terms["decoder"].state_dict())
    rollout(algo, env)
    losses = algo.update()
    assert losses["loss/estimate"] > 0
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert any(
        not torch.equal(value, algo.networks.terms["estimator"].state_dict()[key])
        for key, value in estimator_before.items()
    )
    for key, value in decoder_before.items():
        torch.testing.assert_close(algo.networks.terms["decoder"].state_dict()[key], value)
    # One shared normalizer update, not one update per actor/critic view.
    assert algo.networks.terms["estimator"].obs_normalizer.count == 16


@pytest.mark.parametrize("trains", [["estimator"], ["decoder"], ["estimator", "decoder"]])
def test_exact_loss_gradient_routing(trains: list[str]) -> None:
    """Unselected weights do not train, but remain differentiable toward selected upstream terms."""
    obs = Environment().get_observations()
    manager = NetworkManager(obs, config()["networks"], 2)
    loss = LossManager(
        manager, {"reconstruct": {"trains": trains, "prediction": "decoder", "target": "policy"}}, set(obs.keys())
    )
    joint, _, _ = loss.compute(RolloutStorage.Batch(observations=obs))
    sum(joint).backward()
    for name, model in manager.terms.items():
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()) == (name in trains)


def test_separate_estimator_is_isolated_from_ppo() -> None:
    """PPO cannot update a separately owned estimator through actor or critic inputs."""
    algo, env = make_algorithm(separate=True)
    obs = env.get_observations()
    (algo.actor(obs).sum() + algo.critic(obs).sum()).backward()
    assert all(p.grad is None for p in algo.networks.terms["estimator"].parameters())
    assert any(p.grad is not None for p in algo.networks.terms["actor"].parameters())


def test_joint_estimator_receives_ppo_gradient_without_auxiliary_loss() -> None:
    """A regular dependency receives PPO gradients even with no loss term configured."""
    env = Environment()
    cfg = config()
    cfg["loss"] = {}
    algo = CompositionalPPO.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    algo.actor(env.get_observations()).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in algo.networks.terms["estimator"].parameters())


def test_recurrent_loss_ignores_padding() -> None:
    """Supervision matches only valid rollout samples across staggered episode boundaries."""
    algo, env = make_algorithm("gru")
    rollout(algo, env)
    batch = next(algo.storage.recurrent_mini_batch_generator(2, 1))
    joint, _, _ = algo.losses.compute(batch)
    target = unpad_trajectories(batch.observations["target"], batch.masks)
    prediction = algo.networks.predict("estimator", batch.observations, batch.masks)
    torch.testing.assert_close(joint[0], 0.2 * torch.nn.functional.mse_loss(prediction, target))
    padded = batch.observations.clone()
    padded["target"][~batch.masks] = 1e6
    batch.observations = padded
    changed, _, _ = algo.losses.compute(batch)
    torch.testing.assert_close(changed[0], joint[0])


@pytest.mark.parametrize("recurrent", [None, "gru", "lstm"])
def test_runner_train_save_load_and_jit(tmp_path: Path, recurrent: str | None) -> None:
    """Use upstream OnPolicyRunner directly for training, resume, inference, and export."""
    runner = OnPolicyRunner(Environment(), config(recurrent, separate=True), log_dir=None, device="cpu")
    runner.learn(1)
    path = tmp_path / "checkpoint.pt"
    runner.save(str(path))
    restored = OnPolicyRunner(Environment(), config(recurrent, separate=True), log_dir=None, device="cpu")
    restored.load(str(path))
    assert restored.current_learning_iteration == runner.current_learning_iteration
    for key, value in runner.alg.networks.state_dict().items():
        torch.testing.assert_close(restored.alg.networks.state_dict()[key], value)
    assert restored.alg.losses.optimizers["estimate"].state
    restored.learn(1)
    restored.export_policy_to_jit(str(tmp_path))
    script = torch.jit.load(str(tmp_path / "policy.pt"))
    policy = restored.get_inference_policy()
    if recurrent:
        policy.head.rnn.hidden_state = None
    for _ in range(3):
        obs = Environment().get_observations()[:1]
        torch.testing.assert_close(script(obs["policy"]), policy(obs))
    script.reset()
    policy.reset()
    torch.testing.assert_close(script(obs["policy"]), policy(obs))


@pytest.mark.parametrize("recurrent", [None, "gru", "lstm"])
def test_onnx_export_matches_policy(tmp_path: Path, recurrent: str | None) -> None:
    """Check the entire exported graph independently, including recurrent state outputs."""
    algo, env = make_algorithm(recurrent)
    algo.eval_mode()
    export = algo.actor.as_onnx()
    path = tmp_path / "policy.onnx"
    torch.onnx.export(
        export,
        export.get_dummy_inputs(),
        str(path),
        opset_version=18,
        input_names=export.input_names,
        output_names=export.output_names,
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    obs = env.get_observations()[:1]
    inputs = (obs["policy"], *export.get_dummy_inputs()[1:])
    result = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
        None, {name: value.numpy() for name, value in zip(export.input_names, inputs)}
    )
    torch.testing.assert_close(torch.from_numpy(result[0]), algo.actor(obs), rtol=1e-4, atol=1e-5)
    if recurrent:
        state = algo.actor.get_hidden_state()
        states = state if isinstance(state, tuple) else (state,)
        for exported_state, native_state in zip(result[1:], states):
            torch.testing.assert_close(torch.from_numpy(exported_state), native_state, rtol=1e-4, atol=1e-5)


def test_pretrained_and_frozen_term(tmp_path: Path) -> None:
    """A pretrained shared term preserves its parameters and normalization statistics during PPO."""
    algo, env = make_algorithm()
    rollout(algo, env)
    algo.update()
    path = tmp_path / "pretrained.pt"
    torch.save(algo.save(), path)
    cfg = config()
    cfg["networks"]["estimator"].update(
        frozen=True,
        pretrained_cfg={"path": str(path), "state_dict_key": "network_state_dict", "prefix": "terms.estimator."},
    )
    cfg["loss"] = {}
    restored = CompositionalPPO.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    before = copy.deepcopy(algo.networks.terms["estimator"].state_dict())
    restored.train_mode()
    assert not restored.networks.terms["estimator"].training
    rollout(restored, env)
    restored.update()
    for key, value in before.items():
        torch.testing.assert_close(restored.networks.terms["estimator"].state_dict()[key], value, rtol=0, atol=0)


def test_ownership_change_rejects_optimizer_resume() -> None:
    """Changing separate/joint mode requires explicit weights-only transfer, not silent Adam remapping."""
    source, _ = make_algorithm(separate=True)
    target, _ = make_algorithm()
    with pytest.raises(ValueError, match="Checkpoint"):
        target.load(source.save(), None, True)
    target.load(source.save(), {"networks": True}, True)


@pytest.mark.parametrize("loss_fn", ["mse", "huber", "bce_with_logits"])
def test_builtin_losses(loss_fn: str) -> None:
    """Built-in criteria produce finite, differentiable supervised objectives."""
    algo, env = make_algorithm()
    cfg = config()["loss"]
    cfg["estimate"]["loss_fn"] = loss_fn
    losses = LossManager(algo.networks, cfg, {"policy", "target"})
    joint, _, metrics = losses.compute(RolloutStorage.Batch(observations=env.get_observations()))
    assert torch.isfinite(joint[0]) and metrics["estimate"] >= 0


def test_bad_graph_and_conflicting_optimizers_fail() -> None:
    """Cycles and multiple separate optimizers for one term fail before rollout."""
    obs = Environment().get_observations()
    cfg = config()
    cfg["networks"]["estimator"]["input"] = ["decoder"]
    with pytest.raises(ValueError, match="cycle"):
        NetworkManager(obs, cfg["networks"], 2)
    cfg = config(separate=True)
    cfg["loss"]["duplicate"] = copy.deepcopy(cfg["loss"]["estimate"])
    with pytest.raises(ValueError, match="ownership"):
        LossManager(NetworkManager(obs, cfg["networks"], 2), cfg["loss"], set(obs.keys()))
