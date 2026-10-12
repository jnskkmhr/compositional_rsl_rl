# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility with upstream algorithms, image/recurrent graphs, and IsaacLab configuration."""

from __future__ import annotations

import copy
import runpy
import torch
from pathlib import Path
from tensordict import TensorDict

import pytest
from compositional_rsl_rl.network import GraphModel
from test_composition import Environment, config, rollout

from compositional_rsl_rl import CompositionalPPO, NetworkManager
from rsl_rl.algorithms import PPO


@pytest.mark.parametrize("actor_recurrent,critic_recurrent", [(True, False), (False, True)])
def test_mixed_recurrence(actor_recurrent: bool, critic_recurrent: bool) -> None:
    """Either head can be swapped to an RNN independently of the other."""
    env = Environment()
    cfg = config("gru")
    for name, recurrent in (("actor", actor_recurrent), ("critic", critic_recurrent)):
        if not recurrent:
            term = cfg["networks"][name]
            term["class_name"] = "MLPModel"
            for key in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                term.pop(key)
    algo = CompositionalPPO.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    rollout(algo, env)
    assert all(torch.isfinite(torch.tensor(value)) for value in algo.update().values())


def test_plain_rsl_config_and_checkpoint_remain_compatible() -> None:
    """Selecting the extension with a legacy actor/critic config delegates to ordinary upstream PPO."""
    env = Environment()
    obs = env.get_observations()
    cfg = config()
    cfg.pop("networks")
    cfg.pop("loss")
    cfg["obs_groups"] = {"actor": ["policy"], "critic": ["policy"]}
    cfg["actor"] = {
        "class_name": "MLPModel",
        "hidden_dims": [8],
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0},
    }
    cfg["critic"] = {"class_name": "MLPModel", "hidden_dims": [8]}
    upstream_cfg = copy.deepcopy(cfg)
    upstream_cfg["algorithm"]["class_name"] = "PPO"
    upstream_cfg["multi_gpu"] = None
    source = PPO.construct_algorithm(obs, env, upstream_cfg, "cpu")
    restored = CompositionalPPO.construct_algorithm(obs, env, cfg, "cpu")
    assert type(restored) is PPO
    restored.load(source.save(), None, True)
    torch.testing.assert_close(source.actor(obs), restored.actor(obs))


class ImageEnvironment(Environment):
    """Vector observations plus an image group."""

    def get_observations(self) -> TensorDict:
        """Provide unflattened images for CNN terms."""
        obs = super().get_observations()
        obs["image"] = torch.randn(4, 1, 6, 6)
        return obs


@pytest.mark.parametrize("cnn_head", [False, True])
def test_cnn_and_recurrent_critic(cnn_head: bool) -> None:
    """Both an intermediate CNN and an upstream CNN actor work with recurrent minibatches."""
    env = ImageEnvironment()
    cfg = config("gru")
    cfg["networks"]["actor"] = config()["networks"]["actor"]
    cnn = {
        "class_name": "CNNModel",
        "input": ["image"],
        "output_dim": 4,
        "hidden_dims": [8],
        "cnn_cfg": {"output_channels": [2], "kernel_size": 3},
    }
    if cnn_head:
        cnn["output_dim"] = None
        cnn["distribution_cfg"] = {"class_name": "GaussianDistribution", "init_std": 1.0}
        cfg["networks"]["actor"] = cnn
    else:
        cfg["networks"]["vision"] = cnn
        cfg["networks"]["actor"]["input"].append("vision")
    algo = CompositionalPPO.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    rollout(algo, env)
    assert all(torch.isfinite(torch.tensor(value)) for value in algo.update().values())
    algo.eval_mode()
    export = torch.jit.script(algo.actor.as_jit())
    obs = env.get_observations()
    packed = torch.cat([obs[name].flatten(1) for name in algo.actor.obs_groups], dim=-1)
    torch.testing.assert_close(export(packed), algo.actor(obs))
    assert "target" not in algo.actor.obs_groups


def _mirror(env: object, obs: TensorDict | None, actions: torch.Tensor | None) -> tuple:
    return (
        torch.cat([obs, -obs]) if obs is not None else None,
        torch.cat([actions, -actions]) if actions is not None else None,
    )


def _positive_target_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert (target > 0).all()  # Mirrored targets would be negative.
    return torch.nn.functional.mse_loss(prediction, target)


@pytest.mark.parametrize("separate", [False, True])
def test_symmetry_rnd_and_mixed_precision(separate: bool) -> None:
    """Named losses see original targets alongside upstream symmetry, RND, and autocast."""
    env = Environment()
    cfg = config(separate=separate)
    cfg["loss"]["estimate"]["loss_fn"] = _positive_target_mse
    cfg["obs_groups"] = {"rnd_state": ["policy"]}
    cfg["algorithm"].update(
        use_mixed_precision=True,
        symmetry_cfg={
            "use_data_augmentation": True,
            "use_mirror_loss": True,
            "mirror_loss_coeff": 0.1,
            "data_augmentation_func": _mirror,
        },
        rnd_cfg={"num_outputs": 3, "predictor_hidden_dims": [8], "target_hidden_dims": [8], "weight": 0.1},
    )
    algo = CompositionalPPO.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    rollout(algo, env)
    losses = algo.update()
    assert {"symmetry", "rnd", "loss/estimate"}.issubset(losses)
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())


def test_auxiliary_only_graph_is_pruned_from_export() -> None:
    """The exported actor needs neither targets nor a decoder that only supplies a loss."""
    env = Environment()
    cfg = config()
    cfg["networks"]["decoder"]["input"] = ["target"]
    manager = NetworkManager(env.get_observations(), cfg["networks"], 2)
    actor = GraphModel(manager, "actor")
    assert actor.obs_groups == ["policy"]
    assert "decoder" not in actor.terms
    assert "decoder" in manager.terms


def test_zero_weight_retains_separate_ownership() -> None:
    """Disabling supervision by weight must not accidentally enable PPO gradients into the estimator."""
    env = Environment()
    cfg = config(separate=True)
    cfg["loss"]["estimate"]["weight"] = 0
    algo = CompositionalPPO.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    before = {name: p.detach().clone() for name, p in algo.networks.terms["estimator"].named_parameters()}
    rollout(algo, env)
    losses = algo.update()
    assert "loss/estimate" not in losses
    for name, parameter in algo.networks.terms["estimator"].named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_compile_tracing_preserves_graph_and_gradients() -> None:
    """Dynamo tracing follows the same shared-parameter graph as eager execution."""
    env = Environment()
    cfg = config()
    algo = CompositionalPPO.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    obs = env.get_observations()
    compiled = torch.compile(algo.actor, backend="eager")
    torch.testing.assert_close(compiled(obs), algo.actor(obs))
    compiled(obs).sum().backward()
    assert any(p.grad is not None for p in algo.networks.terms["estimator"].parameters())


def test_actual_isaaclab_config_roundtrip() -> None:
    """The real configclass serializer feeds the extension factory without adapting the runner."""
    pytest.importorskip("isaaclab")
    from isaaclab.utils import to_dict, validate

    example = Path(__file__).parents[1] / "examples" / "isaaclab_g1.py"
    namespace = runpy.run_path(str(example))
    cls = namespace["G1RoughCompositionalPPORunnerCfg"]
    cfg = cls()
    cfg.from_dict({
        "algorithm": {"learning_rate": 0.0005},
        "networks": {"state_estimator": {"hidden_dims": [32, 16]}},
    })
    validate(cfg)
    data = to_dict(cfg)
    assert data["algorithm"]["learning_rate"] == pytest.approx(0.0005)
    assert data["networks"]["state_estimator"]["hidden_dims"] == [32, 16]
    assert cls().networks.state_estimator.hidden_dims == [256, 128]
    assert data["class_name"] == "OnPolicyRunner"
    assert data["algorithm"]["class_name"] == "compositional_rsl_rl:CompositionalPPO"
    assert data["actor"] is None and data["critic"] is None
    obs = TensorDict(
        {
            "policy": torch.randn(4, 6),
            "command": torch.randn(4, 3),
            "critic": torch.randn(4, 8),
            "privileged": torch.randn(4, 5),
            "privileged_state": torch.randn(4, 3),
        },
        batch_size=[4],
    )
    algo = CompositionalPPO.construct_algorithm(obs, Environment(), data, "cpu")
    assert algo.actor(obs).shape == (4, 2)
    assert cls().networks.actor.input == ["policy", "state_estimator", "command"]
    cfg.networks.actor.input.append("extra")
    assert "extra" not in cls().networks.actor.input
