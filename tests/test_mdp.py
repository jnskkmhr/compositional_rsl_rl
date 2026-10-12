# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable term masking, custom import paths, training, and deployment contracts."""

from __future__ import annotations

import runpy
import torch
from dataclasses import asdict
from pathlib import Path
from tensordict import TensorDict

import onnxruntime
import pytest
from compositional_rsl_rl.mdp.network import PointNetEncoder, PointNetNetworkCfg, TransformerEncoder, TransformerNetworkCfg
from compositional_rsl_rl.network import TensorNetworkCfg
from test_composition import Environment, config, rollout

from compositional_rsl_rl import CompositionalPPO


class _PointEnvironment(Environment):
    def get_observations(self) -> TensorDict:
        obs = super().get_observations()
        points = torch.randn(self.num_envs, 5, 4)
        points[..., 3] = 1
        points[:, -1, :3] = float("nan")
        points[:, -1, 3] = 0
        obs["points"] = points
        return obs


def _network_cfg(kind: str) -> dict:
    kwargs = {"input": ["points"], "output_dim": 3, "feature_dim": 8, "hidden_dims": [8]}
    if kind == "pointnet":
        return asdict(PointNetNetworkCfg(**kwargs, pointnet_hidden_dims=[8]))
    return asdict(TransformerNetworkCfg(**kwargs, num_heads=2, num_layers=1, feedforward_dim=16, validity_channel=True))


@pytest.mark.parametrize("kind", ["pointnet", "transformer"])
def test_masking_and_set_invariance(kind: str) -> None:
    """Invalid NaN padding and empty inputs must not contaminate features or gradients."""
    if kind == "pointnet":
        encoder = PointNetEncoder([8], 8, "elu")
    else:
        encoder = TransformerEncoder((5, 4), 8, 2, 1, 16, validity_channel=True, use_positional_encoding=False)
    points = torch.randn(3, 5, 4)
    points[..., 3] = 1
    points[0, :, 3] = 0
    points[1, -2:, 3] = 0
    invalid = points[..., 3] == 0
    points[invalid] = torch.tensor([float("nan"), float("nan"), float("nan"), 0.0])
    points.requires_grad_()
    output = encoder(points)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[0], torch.zeros(8))
    torch.testing.assert_close(output, encoder(points[:, torch.tensor([4, 2, 0, 3, 1])]))
    output.square().sum().backward()
    assert torch.isfinite(points.grad).all()
    torch.testing.assert_close(points.grad[invalid], torch.zeros_like(points.grad[invalid]))


def test_transformer_positions_and_validation() -> None:
    """Positional embeddings distinguish token order; invalid attention dimensions fail early."""
    torch.manual_seed(17)
    encoder = TransformerEncoder((5, 3), 8, 2, 1, 16)
    tokens = torch.randn(2, 5, 3)
    assert not torch.allclose(encoder(tokens), encoder(tokens.flip(1)))
    with pytest.raises(ValueError, match="divisible"):
        TransformerEncoder((5, 3), 7, 2)


@pytest.mark.parametrize("kind", ["pointnet", "transformer"])
@pytest.mark.parametrize("recurrent", [None, "gru"])
@pytest.mark.parametrize("separate", [False, True])
def test_structured_term_training_and_export(kind: str, recurrent: str | None, separate: bool) -> None:
    """Structured terms learn under either optimizer and export inside feed-forward/recurrent graphs."""
    env = _PointEnvironment()
    cfg = config(recurrent, separate)
    cfg["networks"]["estimator"] = _network_cfg(kind)
    cfg["loss"]["estimate"]["loss_fn"] = "compositional_rsl_rl.mdp.loss:cosine_distance"
    algo = CompositionalPPO.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    before = [p.detach().clone() for p in algo.networks.terms["estimator"].parameters()]
    rollout(algo, env)
    metrics = algo.update()
    assert metrics["loss/estimate"] >= 0
    assert any(not torch.equal(a, b) for a, b in zip(before, algo.networks.terms["estimator"].parameters()))
    algo.eval_mode()
    obs = env.get_observations()[:1]
    algo.actor.reset()
    expected = algo.actor(obs)
    packed = torch.cat([obs[name].flatten(start_dim=1) for name in algo.actor.obs_groups], dim=-1)
    exported = torch.jit.script(algo.actor.as_jit())
    exported.reset()
    torch.testing.assert_close(exported(packed), expected)


@pytest.mark.parametrize("kind", ["pointnet", "transformer"])
def test_structured_term_onnx(kind: str, tmp_path: Path) -> None:
    """Verify the complete exported graph with independent ONNX Runtime execution."""
    env = _PointEnvironment()
    cfg = config()
    cfg["networks"]["estimator"] = _network_cfg(kind)
    algo = CompositionalPPO.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    algo.eval_mode()
    export = algo.actor.as_onnx()
    path = tmp_path / "policy.onnx"
    torch.onnx.export(
        export,
        export.get_dummy_inputs(),
        path,
        input_names=export.input_names,
        output_names=export.output_names,
        opset_version=18,
        dynamo=True,
    )
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    obs = env.get_observations()[:1]
    packed = torch.cat([obs[name].flatten(start_dim=1) for name in algo.actor.obs_groups], dim=-1)
    result = session.run(None, {"obs": packed.numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(result), algo.actor(obs), rtol=1e-4, atol=1e-5)


def test_external_network_needs_no_registration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Users can add a tensor encoder in their own package without modifying either manager."""
    (tmp_path / "user_terms.py").write_text(
        "from torch import nn\n"
        "from compositional_rsl_rl.network import TensorNetwork\n"
        "class MyEncoder(TensorNetwork):\n"
        "    def build_encoder(self, input_shape, feature_dim, activation):\n"
        "        return nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(input_shape[0], feature_dim))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    env = Environment()
    cfg = config()
    cfg["networks"]["estimator"] = asdict(
        TensorNetworkCfg(class_name="user_terms:MyEncoder", input=["policy"], output_dim=3, feature_dim=8)
    )
    algo = CompositionalPPO.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    rollout(algo, env)
    assert torch.isfinite(torch.tensor(algo.update()["loss/estimate"]))


@pytest.mark.parametrize("kind", ["pointnet", "transformer"])
def test_perception_example_defaults(kind: str) -> None:
    """The published example must construct and train with its default encoder/readout settings."""
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "examples/perception.py"))
    cfg = asdict(namespace["make_config"](kind))
    cfg["num_steps_per_env"] = 4
    cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=2)
    cfg["loss"]["perception_loss"]["target"] = "target"
    env = _PointEnvironment()
    algo = CompositionalPPO.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    rollout(algo, env)
    assert torch.isfinite(torch.tensor(algo.update()["loss/perception_loss"]))
