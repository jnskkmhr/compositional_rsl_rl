"""Holosoma learner integration: normalization, ownership, replay and actor export."""

import copy
from dataclasses import asdict
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from rsl_rl.runners import OffPolicyRunner
from tensordict import TensorDict

from compositional_rsl_rl import CompositionalFastSAC
from compositional_rsl_rl.algorithm import CompositionalFastSACAlgorithmCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg
from compositional_rsl_rl.runner import CompositionalFastSACRunnerCfg


def configuration():
    return {
        "algorithm": asdict(CompositionalFastSACAlgorithmCfg(num_bins=11, action_scale=[2.0, 0.5])),
        "networks": {
            "actor": asdict(
                ActorHeadCfg(
                    input=["policy"],
                    hidden_dims=[8, 6],
                    layer_norm=True,
                    activation="swish",
                    distribution_cfg=ActorHeadCfg.SquashedGaussianDistributionCfg(
                        log_std_min=-5.0,
                        log_std_max=0.0,
                        zero_init_output=True,
                    ),
                )
            ),
            "critic": asdict(
                CriticHeadCfg(
                    input=["critic"],
                    hidden_dims=[8, 6],
                    layer_norm=True,
                    num_ensembles=2,
                    activation="swish",
                    distribution_cfg=CriticHeadCfg.CategoricalValueDistributionCfg(
                        num_bins=11,
                        value_min=-20.0,
                        value_max=20.0,
                    ),
                )
            ),
        },
    }


def batch():
    obs = TensorDict({"policy": torch.randn(8, 3), "critic": torch.randn(8, 4)}, [8])
    return TensorDict(
        {
            "observations": obs,
            "next_observations": obs + 2,
            "actions": torch.randn(8, 2),
            "rewards": torch.randn(8),
            "terminated": torch.zeros(8, dtype=torch.bool),
            "bootstrap_discount": torch.full((8,), 0.97),
        },
        [8],
    )


def test_fast_sac_normalization_schedule_and_checkpoint():
    data = batch()
    obs = data["observations"]
    cfg = configuration()
    alg = CompositionalFastSAC.construct_algorithm(obs, SimpleNamespace(num_actions=2), cfg, "cpu")
    assert alg.normalizers["actor"].count == 0
    first = alg.act(obs)
    assert not first.equal(alg.act(obs))  # IID exploration, not repeated zeta noise.
    assert alg.normalizers["actor"].count == 0  # Collection does not fit statistics.
    assert first[:, 0].abs().max() <= 2 and first[:, 1].abs().max() <= 0.5
    metrics = list(alg.update_batches([data.clone() for _ in range(8)]))
    assert [i for i, value in enumerate(metrics) if "actor/loss" in value] == [1, 5]
    assert all("temperature/loss" in value for value in metrics)
    assert alg.normalizers["actor"].count == 128  # 8 batches, current + next observations.
    torch.testing.assert_close(alg.normalizers["actor"].mean, obs["policy"].mean(0) + 1)
    assert alg.normalizers["critic"].mean.shape == (4,)  # Actions are excluded.
    assert isinstance(alg.optimizers["actor"], torch.optim.AdamW)
    assert alg.optimizers["actor"].param_groups[0]["betas"] == (0.9, 0.95)
    state = copy.deepcopy(alg.save())
    expected_metrics = alg.update(data)
    expected = copy.deepcopy(alg.save())
    restored = CompositionalFastSAC.construct_algorithm(obs, SimpleNamespace(num_actions=2), cfg, "cpu")
    restored.load(state)
    actual_metrics = restored.update(data)
    for key in expected_metrics:
        torch.testing.assert_close(actual_metrics[key], expected_metrics[key])
    for role, model in restored.models.items():
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected["models"][role][key])
    torch.testing.assert_close(restored.get_policy()(obs), restored.deterministic_actions(obs))
    assert restored.normalizers["actor"].count == 144
    with pytest.raises(ValueError, match="FastSAC checkpoint"):
        restored.load({key: value for key, value in state.items() if key != "fast_sac_version"})


def test_actor_frequency_one_and_disabled_temperature():
    data = batch()
    cfg = configuration()
    cfg["algorithm"].update(actor_update_period=1, use_autotune=False)
    alg = CompositionalFastSAC.construct_algorithm(data["observations"], SimpleNamespace(num_actions=2), cfg, "cpu")
    original = alg.temperature.log_temp.detach().clone()
    metrics = list(alg.update_batches([data.clone(), data.clone()]))
    assert all("actor/loss" in value and "temperature/loss" not in value for value in metrics)
    torch.testing.assert_close(alg.temperature.log_temp, original)


def test_fast_sac_runner_final_observation_and_exports(tmp_path):
    obs = batch()["observations"]

    class Env:
        num_envs = 8
        num_actions = 2
        device = "cpu"
        max_episode_length = 3
        episode_length_buf = torch.zeros(8, dtype=torch.long)
        cfg: ClassVar[dict] = {}

        def get_observations(self):
            return obs

        def step(self, actions):
            self.episode_length_buf += 1
            done = self.episode_length_buf % 3 == 0
            return (
                obs,
                -actions.square().sum(-1),
                done,
                {
                    "terminated": torch.zeros_like(done),
                    "truncated": done,
                    "final_obs": obs + 10,
                    "final_obs_mask": done,
                },
            )

    cfg = configuration()
    runner_cfg = asdict(
        CompositionalFastSACRunnerCfg(
            networks=cfg["networks"],
            algorithm=CompositionalFastSACAlgorithmCfg(
                **{key: value for key, value in cfg["algorithm"].items() if key != "class_name"}
            ),
            learning_starts=0,
            replay_steps_per_env=8,
            replay_max_bytes=1_000_000,
            batch_size=8,
            updates_per_step=2,
        )
    )
    runner = OffPolicyRunner(Env(), runner_cfg)
    runner.learn(4)
    assert runner.alg.update_step == 6
    assert runner.replay.capacity == 64
    torch.testing.assert_close(runner.replay.storage[16:24]["next_observations", "policy"], obs["policy"] + 10)
    expected = runner.get_inference_policy()(obs[:2])
    runner.export_policy_to_jit(str(tmp_path))
    runner.export_policy_to_onnx(str(tmp_path))
    inputs = obs[:2]["policy"]
    torch.testing.assert_close(torch.jit.load(str(tmp_path / "policy.pt"))(inputs), expected)
    import onnxruntime

    session = onnxruntime.InferenceSession(str(tmp_path / "policy.onnx"), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"obs": inputs.numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(actual), expected, atol=1e-5, rtol=1e-5)
