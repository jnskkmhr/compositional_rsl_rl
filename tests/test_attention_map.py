# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AME query dependence, sharing, normalization, training, and deployment contracts."""

from __future__ import annotations

import copy
import runpy
import torch
from dataclasses import asdict
from pathlib import Path
from tensordict import TensorDict

import onnxruntime
import pytest
from compositional_rsl_rl.mdp.network import AttentionMapEncoder, AttentionMapNetwork, AttentionMapNetworkCfg
from test_composition import Environment, config, rollout

from compositional_rsl_rl import CompositionalPPO, NetworkManager
from rsl_rl.runners import OnPolicyRunner


class _MapEnvironment(Environment):
    def get_observations(self) -> TensorDict:
        obs = super().get_observations()
        obs["critic"] = torch.randn(self.num_envs, 9)
        obs["elevation_map"] = torch.randn(self.num_envs, 4, 6, 3)
        return obs


def _config(recurrent: str | None = None, attach_global: bool = False, separate: bool = False) -> dict:
    cfg = config(recurrent, separate)
    cfg["networks"].pop("decoder")
    cfg["networks"]["critic"]["input"] = ["critic", "estimator"]
    cfg["networks"]["estimator"] = asdict(
        AttentionMapNetworkCfg(
            input=["elevation_map", "policy"],
            output_dim=3,
            feature_dim=8,
            num_heads=2,
            cnn_channels=4,
            hidden_dims=[8],
            attach_global=attach_global,
            global_hidden_dims=[8],
            obs_normalization=True,
        )
    )
    return cfg


def test_cross_attention_matches_pytorch() -> None:
    """The exportable attention implementation must implement PyTorch's single-query MHA exactly."""
    torch.manual_seed(19)
    encoder = AttentionMapEncoder(6, 8, 2, 4, cnn_normalization="none").eval()
    reference = torch.nn.MultiheadAttention(8, 2, batch_first=True).eval()
    actual = encoder.attention
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat([actual.query.weight, actual.key.weight, actual.value.weight]))
        reference.in_proj_bias.copy_(torch.cat([actual.query.bias, actual.key.bias, actual.value.bias]))
        reference.out_proj.load_state_dict(actual.output.state_dict())
    tokens = torch.randn(3, 6, 8)
    query = torch.randn(3, 8)
    expected, expected_weights = reference(query[:, None], tokens, tokens, average_attn_weights=False)
    features, weights = actual(query, tokens)
    torch.testing.assert_close(features, expected[:, 0])
    torch.testing.assert_close(weights, expected_weights[:, :, 0])


@pytest.mark.parametrize("downsample", [False, True])
@pytest.mark.parametrize("attach_global", [False, True])
def test_map_query_dependence_and_weights(downsample: bool, attach_global: bool) -> None:
    """Map features depend on state queries and XYZ input; weights form per-head spatial distributions."""
    torch.manual_seed(23)
    encoder = AttentionMapEncoder(6, 8, 2, 4, downsample, "none", attach_global, [8]).eval()
    xyz = torch.randn(2, 4, 6, 3, requires_grad=True)
    state = torch.randn(2, 6, requires_grad=True)
    features, weights = encoder.encode(xyz, state)
    assert features.shape == (2, 16 if attach_global else 8)
    assert weights.shape == (2, 2, 2, 3) if downsample else weights.shape == (2, 2, 4, 6)
    torch.testing.assert_close(weights.sum(dim=(-2, -1)), torch.ones(2, 2))
    assert not torch.allclose(features, encoder(xyz, state + 2))
    features.square().sum().backward()
    assert torch.isfinite(xyz.grad).all() and torch.count_nonzero(xyz.grad)
    assert torch.isfinite(state.grad).all() and torch.count_nonzero(state.grad)


def test_flattened_map_and_input_order() -> None:
    """The map may occur between query groups, with structured or explicitly shaped flat storage."""
    obs = _MapEnvironment().get_observations()
    obs["command"] = torch.randn(4, 2)
    groups = {"term": ["policy", "elevation_map", "command"]}
    term = AttentionMapNetwork(obs, groups, "term", 8, feature_dim=8, num_heads=2, cnn_normalization="none")
    flat = obs.clone()
    flat["elevation_map"] = flat["elevation_map"].flatten(1)
    other = AttentionMapNetwork(
        flat, groups, "term", 8, map_shape=(4, 6, 3), feature_dim=8, num_heads=2, cnn_normalization="none"
    )
    other.load_state_dict(term.state_dict())
    term.eval()
    other.eval()
    packed = torch.cat([obs[name].flatten(1) for name in groups["term"]], dim=-1)
    torch.testing.assert_close(term(obs), other(flat))
    torch.testing.assert_close(torch.jit.script(term.as_jit())(packed), term(obs))
    torch.testing.assert_close(torch.jit.script(other.as_jit())(packed), other(flat))
    before = copy.deepcopy(term.state_dict())
    assert term.attention_weights(obs).shape == (4, 2, 2, 3)
    for key, value in before.items():
        torch.testing.assert_close(term.state_dict()[key], value, rtol=0, atol=0)
    term.train()
    with pytest.raises(RuntimeError, match="eval"):
        term.attention_weights(obs)


@pytest.mark.parametrize("recurrent", [None, "gru", "lstm"])
@pytest.mark.parametrize("separate", [False, True])
def test_ame_training_and_actor_export(recurrent: str | None, separate: bool) -> None:
    """Shared encoders train with either owner, including recurrent batches and actor-only export."""
    env = _MapEnvironment()
    cfg = _config(recurrent, attach_global=True, separate=separate)
    algo = CompositionalPPO.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    term = algo.networks.terms["estimator"]
    assert algo.actor.terms["estimator"] is algo.critic.terms["estimator"]
    optimizers = [algo.optimizer, *algo.losses.optimizers.values()]
    ids = [id(p) for opt in optimizers for group in opt.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    before = [p.detach().clone() for p in term.parameters()]
    rollout(algo, env)
    assert torch.isfinite(torch.tensor(algo.update()["loss/estimate"]))
    assert any(not torch.equal(a, b) for a, b in zip(before, term.parameters()))
    assert term.obs_normalizer.count == 16  # Shared query normalizer updated once, never on XYZ coordinates.
    assert term.obs_dim == 6
    algo.eval_mode()
    obs = env.get_observations().exclude("critic", "target")
    if recurrent:
        obs = obs[:1]
    algo.actor.reset()
    expected = algo.actor(obs)
    assert set(algo.actor.obs_groups) == {"policy", "elevation_map"}
    packed = torch.cat([obs[name].flatten(1) for name in algo.actor.obs_groups], dim=-1)
    exported = torch.jit.script(algo.actor.as_jit())
    exported.reset()
    torch.testing.assert_close(exported(packed), expected)


@pytest.mark.parametrize("attach_global", [False, True])
def test_ame_onnx(attach_global: bool, tmp_path: Path) -> None:
    """Run the complete exported actor independently with ONNX Runtime."""
    env = _MapEnvironment()
    algo = CompositionalPPO.construct_algorithm(
        env.get_observations(), env, _config(attach_global=attach_global), "cpu"
    )
    algo.eval_mode()
    export = algo.actor.as_onnx()
    path = tmp_path / "actor.onnx"
    torch.onnx.export(
        export,
        export.get_dummy_inputs(),
        path,
        opset_version=18,
        dynamo=True,
        input_names=export.input_names,
        output_names=export.output_names,
    )
    obs = env.get_observations()[:1].exclude("critic", "target")
    packed = torch.cat([obs[name].flatten(1) for name in algo.actor.obs_groups], dim=-1)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"obs": packed.numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(actual), algo.actor(obs), rtol=1e-4, atol=1e-5)


def test_ame_checkpoint_and_freeze(tmp_path: Path) -> None:
    """Runner resume restores AME state; frozen transfer preserves CNN buffers and query statistics."""
    env = _MapEnvironment()
    cfg = _config()
    runner = OnPolicyRunner(env, cfg, log_dir=None, device="cpu")
    runner.learn(num_learning_iterations=1)
    checkpoint = tmp_path / "model.pt"
    runner.save(str(checkpoint))
    restored = OnPolicyRunner(env, _config(), log_dir=None, device="cpu")
    restored.load(str(checkpoint))
    for key, value in runner.alg.networks.state_dict().items():
        torch.testing.assert_close(restored.alg.networks.state_dict()[key], value, rtol=0, atol=0)
    frozen_cfg = _config()
    frozen_cfg["networks"]["estimator"].update(
        frozen=True,
        pretrained_cfg={"path": str(checkpoint), "state_dict_key": "network_state_dict", "prefix": "terms.estimator."},
    )
    frozen_cfg["loss"] = {}
    frozen = OnPolicyRunner(env, frozen_cfg, log_dir=None, device="cpu")
    before = copy.deepcopy(frozen.alg.networks.terms["estimator"].state_dict())
    frozen.learn(num_learning_iterations=1)
    for key, value in before.items():
        torch.testing.assert_close(frozen.alg.networks.terms["estimator"].state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"map_input": "missing"}, "map_input"),
        ({"input": ["elevation_map"]}, "query input"),
        ({"input": ["policy", "elevation_map", "policy"]}, "unique"),
        ({"map_shape": (6, 4, 3)}, "does not match"),
        ({"num_heads": 3}, "divisible"),
        ({"cnn_normalization": "unknown"}, "cnn_normalization"),
        ({"hidden_dims": []}, "output_dim"),
    ],
)
def test_invalid_ame_configs(changes: dict, message: str) -> None:
    """Reject ambiguous input layouts and incompatible feature dimensions before rollout."""
    cfg = _config()
    cfg["networks"]["estimator"].update(changes)
    with pytest.raises(ValueError, match=message):
        NetworkManager(_MapEnvironment().get_observations(), cfg["networks"], 2)


def test_ame_example_without_auxiliary_loss() -> None:
    """The published no-readout configuration builds without supervision targets."""
    example = Path(__file__).parents[1] / "examples/attention_map.py"
    cfg = asdict(runpy.run_path(str(example))["make_config"]())
    cfg["num_steps_per_env"] = 4
    cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=2)
    cfg["networks"]["map_encoder"]["map_shape"] = (4, 6, 3)
    env = _MapEnvironment()
    algo = CompositionalPPO.construct_algorithm(env.get_observations().exclude("target"), env, cfg, "cpu")
    before = algo.networks.terms["map_encoder"].encoder.state_projection.weight.detach().clone()
    rollout(algo, env)
    losses = algo.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert not algo.losses.terms
    assert not torch.equal(before, algo.networks.terms["map_encoder"].encoder.state_projection.weight)


def test_isaaclab_ame_config() -> None:
    """The complete IsaacLab example supports serialization and nested encoder overrides."""
    pytest.importorskip("isaaclab")
    from isaaclab.utils import to_dict, validate

    example = Path(__file__).parents[1] / "examples/isaaclab_attention_map.py"
    cls = runpy.run_path(str(example))["MyCompositionalPPORunnerCfg"]
    cfg = cls()
    cfg.from_dict({"networks": {"map_encoder": {"attach_global": True, "output_dim": 128}}})
    validate(cfg)
    assert not cls().networks.map_encoder.attach_global
    env = _MapEnvironment()
    obs = env.get_observations().exclude("target")
    obs["elevation_map"] = torch.randn(4, 33, 21, 3)
    algo = CompositionalPPO.construct_algorithm(obs, env, to_dict(cfg), "cpu")
    algo.eval_mode()
    assert algo.actor(obs).shape == (4, 2)
    assert algo.networks.terms["map_encoder"].attention_weights(obs).shape == (4, 16, 17, 11)
