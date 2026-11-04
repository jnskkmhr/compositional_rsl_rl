# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multiple checkpoint teachers supervise explicit student actions during real PPO updates."""

import copy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from compositional_rsl_rl import CompositionalPPO, NetworkManager
from compositional_rsl_rl.loss import DistillationLossCfg, LossManager
from compositional_rsl_rl.network import TeacherPolicyCfg
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from test_composition import Environment, rollout


class DistillationEnvironment(Environment):
    """Student sees proprioception; teachers receive distinct observation groups."""

    def __init__(self) -> None:
        super().__init__()
        self.num_actions = 4

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {"policy": torch.ones(4, 3), "lower_obs": torch.ones(4, 6), "upper_obs": torch.ones(4, 5)},
            batch_size=[4],
        )


def teacher_config(tmp_path: Path, obs: TensorDict, group: str, actions: list[float]) -> dict:
    """Save a real PPO actor checkpoint, including normalization and Gaussian parameters."""
    distribution = {"class_name": "GaussianDistribution", "init_std": 0.7}
    teacher = MLPModel(
        obs,
        {"actor": [group]},
        "actor",
        len(actions),
        hidden_dims=[8],
        obs_normalization=True,
        distribution_cfg=distribution,
    )
    teacher.update_normalization(obs)
    with torch.no_grad():
        teacher.mlp[-1].weight.zero_()
        teacher.mlp[-1].bias.copy_(torch.tensor(actions))
    path = tmp_path / f"{group}.pt"
    torch.save({"actor_state_dict": teacher.state_dict()}, path)
    return asdict(
        TeacherPolicyCfg(
            input=[group],
            output_dim=len(actions),
            hidden_dims=[8],
            obs_normalization=True,
            distribution_cfg=distribution,
            pretrained_cfg={"path": str(path), "state_dict_key": "actor_state_dict"},
        )
    )


def configuration(tmp_path: Path, obs: TensorDict) -> dict:
    return {
        "num_steps_per_env": 4,
        "networks": {
            "actor": {
                "class_name": "MLPModel",
                "input": ["policy"],
                "hidden_dims": [8],
                "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.2},
            },
            "critic": {"class_name": "MLPModel", "input": ["policy"], "hidden_dims": [8]},
            "lower_teacher": teacher_config(tmp_path, obs, "lower_obs", [1.0, 2.0]),
            "upper_teacher": teacher_config(tmp_path, obs, "upper_obs", [-1.0]),
        },
        "loss": {
            "lower": asdict(
                DistillationLossCfg(
                    target="lower_teacher", prediction_indices=[2, 0], target_indices=[1, 0], weight=2.0
                )
            ),
            "upper": asdict(DistillationLossCfg(target="upper_teacher", prediction_indices=[1])),
        },
        "algorithm": {
            "class_name": "compositional_rsl_rl:CompositionalPPO",
            "schedule": "fixed",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "learning_rate": 0.02,
            "entropy_coef": 0.0,
            "value_loss_coef": 1.0,
        },
    }


def test_teacher_checkpoints_action_mapping_and_gradient_ownership(tmp_path: Path) -> None:
    """Ordered mappings target the correct joints; unsupervised actions and teachers get no distillation gradient."""
    env = DistillationEnvironment()
    obs = env.get_observations()
    cfg = configuration(tmp_path, obs)
    manager = NetworkManager(obs, cfg["networks"], env.num_actions)
    actor = manager.terms["actor"]
    with torch.no_grad():
        actor.mlp[-1].weight.zero_()
        actor.mlp[-1].bias.zero_()
    losses = LossManager(manager, cfg["loss"], set(obs.keys()))
    joint, separate, metrics = losses.compute(RolloutStorage.Batch(observations=obs))
    assert not separate
    assert metrics == {"lower": 2.5, "upper": 1.0}
    sum(joint).backward()
    # 2 * mean((a[[2,0]] - [2,1])**2) + (a[1]+1)**2.
    torch.testing.assert_close(actor.mlp[-1].bias.grad, torch.tensor([-2.0, 2.0, -4.0, 0.0]))
    for name in ("lower_teacher", "upper_teacher"):
        teacher = manager.terms[name]
        saved = torch.load(cfg["networks"][name]["pretrained_cfg"]["path"], weights_only=True)["actor_state_dict"]
        for key, value in saved.items():
            torch.testing.assert_close(teacher.state_dict()[key], value, rtol=0, atol=0)
        assert not teacher.training and all(p.grad is None for p in teacher.parameters())
    assert manager.observation_groups("actor") == ["policy"]


def test_ppo_distillation_learns_both_teachers_and_resumes_without_teacher_files(tmp_path: Path) -> None:
    """Both losses improve through PPO, teachers remain frozen, and full checkpoints include their weights."""
    torch.manual_seed(1)
    env = DistillationEnvironment()
    obs = env.get_observations()
    cfg = configuration(tmp_path, obs)
    algo = CompositionalPPO.construct_algorithm(obs, env, cfg, "cpu")
    algo.train_mode()
    frozen = {n: copy.deepcopy(algo.networks.terms[n].state_dict()) for n in ("lower_teacher", "upper_teacher")}
    critic_before = copy.deepcopy(algo.networks.terms["critic"].state_dict())
    _, _, initial = algo.losses.compute(RolloutStorage.Batch(observations=obs))
    for _ in range(20):
        rollout(algo, env)
        metrics = algo.update()
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    _, _, final = algo.losses.compute(RolloutStorage.Batch(observations=obs))
    assert final["lower"] < initial["lower"] * 0.5
    assert final["upper"] < initial["upper"] * 0.5
    assert "loss/lower" in metrics and "loss/upper" in metrics and "surrogate" in metrics
    assert any(not torch.equal(v, algo.networks.terms["critic"].state_dict()[k]) for k, v in critic_before.items())
    for name, state in frozen.items():
        for key, value in state.items():
            torch.testing.assert_close(algo.networks.terms[name].state_dict()[key], value, rtol=0, atol=0)
    checkpoint = algo.save()
    checkpoint["iter"] = 20
    path = tmp_path / "student.pt"
    torch.save(checkpoint, path)
    for name in frozen:
        Path(cfg["networks"][name]["pretrained_cfg"]["path"]).unlink()
        cfg["networks"][name]["pretrained_cfg"] = None
    restored = CompositionalPPO.construct_algorithm(obs, env, cfg, "cpu")
    restored.load(torch.load(path, weights_only=True), None, True)
    torch.testing.assert_close(restored.actor(obs), algo.actor(obs))
    # Export consumes only the student's inputs, with no teacher observation groups.
    exported = torch.jit.script(restored.actor.as_jit())
    torch.testing.assert_close(exported(obs["policy"]), restored.actor(obs))
    rollout(restored, env)
    assert "loss/upper" in restored.update()
    # A full optimizer resume must not silently change which joints a teacher supervises.
    cfg["loss"]["lower"]["prediction_indices"] = [0, 2]
    remapped = CompositionalPPO.construct_algorithm(obs, env, cfg, "cpu")
    with pytest.raises(ValueError, match="loss_bindings"):
        remapped.load(checkpoint, None, True)


@pytest.mark.parametrize("selection", [[], [0, 0], [-1], [True], [0.5]])
def test_invalid_action_mappings_fail_early(tmp_path: Path, selection: list) -> None:
    """Reject ambiguous/invalid action maps before collecting rollout data."""
    env = DistillationEnvironment()
    obs = env.get_observations()
    cfg = configuration(tmp_path, obs)
    cfg["loss"]["lower"]["prediction_indices"] = selection
    with pytest.raises(ValueError, match="unique nonnegative integer"):
        CompositionalPPO.construct_algorithm(obs, env, cfg, "cpu")


def test_distillation_rejects_trainable_teacher_dependencies(tmp_path: Path) -> None:
    """Freezing only the final teacher head must not allow its targets to drift through an encoder."""
    env = DistillationEnvironment()
    obs = env.get_observations()
    cfg = configuration(tmp_path, obs)
    cfg["networks"]["teacher_encoder"] = {
        "class_name": "MLPModel",
        "input": ["lower_obs"],
        "output_dim": 6,
        "hidden_dims": [8],
    }
    cfg["networks"]["lower_teacher"]["input"] = ["teacher_encoder"]
    with pytest.raises(ValueError, match="fully frozen teacher"):
        CompositionalPPO.construct_algorithm(obs, env, cfg, "cpu")


def test_out_of_range_mapping_reports_loss_name(tmp_path: Path) -> None:
    """An action index cannot silently select an environment or wrap around the output width."""
    env = DistillationEnvironment()
    obs = env.get_observations()
    cfg = configuration(tmp_path, obs)
    cfg["loss"]["lower"]["target_indices"] = [2, 0]
    algo = CompositionalPPO.construct_algorithm(obs, env, cfg, "cpu")
    with pytest.raises(ValueError, match="lower.*target_indices"):
        algo.losses.compute(RolloutStorage.Batch(observations=obs))


def test_recurrent_student_distillation_handles_episode_padding(tmp_path: Path) -> None:
    """Teacher actions align with valid recurrent rollout samples across staggered resets."""
    env = DistillationEnvironment()
    cfg = configuration(tmp_path, env.get_observations())
    cfg["networks"]["actor"].update(class_name="RNNModel", rnn_type="gru", rnn_hidden_dim=8, rnn_num_layers=1)
    algo = CompositionalPPO.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    rollout(algo, env)
    batch = next(algo.storage.recurrent_mini_batch_generator(1, 1))
    joint, _, _ = algo.losses.compute(batch)
    prediction = algo.networks.predict("actor", batch.observations, batch.masks, batch.hidden_states[0])
    expected = ((prediction[..., [2, 0]] - torch.tensor([2.0, 1.0])) ** 2).mean() * 2
    expected += ((prediction[..., 1] + 1.0) ** 2).mean()
    torch.testing.assert_close(sum(joint), expected)
    assert "loss/lower" in algo.update()


def _distributed_worker(rank: int, store: str, cfg: dict) -> None:
    """Exercise teacher broadcast and student gradient reduction through a real distributed update."""
    import datetime

    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        store=torch.distributed.FileStore(store, 2),
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=45),
    )
    try:
        torch.manual_seed(50 + rank)
        env = DistillationEnvironment()
        cfg["multi_gpu"] = {"global_rank": rank, "world_size": 2}
        algo = CompositionalPPO.construct_algorithm(env.get_observations(), env, cfg, "cpu")
        # Deliberately different frozen targets must also be synchronized at startup.
        with torch.no_grad():
            algo.networks.terms["lower_teacher"].mlp[-1].bias.add_(rank)
        algo.broadcast_parameters()
        frozen = copy.deepcopy(algo.networks.terms["lower_teacher"].state_dict())
        rollout(algo, env)
        losses = algo.update()
        assert "loss/lower" in losses and "loss/upper" in losses
        for parameter in algo.networks.parameters():
            values = [torch.empty_like(parameter) for _ in range(2)]
            torch.distributed.all_gather(values, parameter)
            torch.testing.assert_close(values[0], values[1], atol=0, rtol=0)
        for key, value in frozen.items():
            torch.testing.assert_close(algo.networks.terms["lower_teacher"].state_dict()[key], value, atol=0, rtol=0)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="PyTorch Gloo is unavailable")
def test_two_worker_distillation(tmp_path: Path) -> None:
    """Multiple teachers remain frozen and students agree after gradients from independent workers are reduced."""
    cfg = configuration(tmp_path, DistillationEnvironment().get_observations())
    torch.multiprocessing.spawn(_distributed_worker, args=(str(tmp_path / "store"), cfg), nprocs=2, join=True)


def test_loss_schedule_transfers_behavior_cloning_to_whole_ppo_and_resumes(tmp_path: Path) -> None:
    """A zero PPO weight disables value regression too; later updates train the actor and critic together."""
    torch.manual_seed(7)
    env = DistillationEnvironment()
    obs = env.get_observations()
    cfg = configuration(tmp_path, obs)
    cfg["algorithm"]["curriculum_cfg"] = {
        name: {"type": "linear", "params": {"start_step": 0, "end_step": 8, "start_value": start, "end_value": end}}
        for name, start, end in [
            ("ppo_loss_weight", 0.0, 1.0),
            ("loss.lower.weight", 2.0, 0.0),
            ("loss.upper.weight", 1.0, 0.0),
        ]
    }
    algo = CompositionalPPO.construct_algorithm(obs, env, cfg, "cpu")
    actor = algo.networks.terms["actor"]
    actor_before = actor.mlp[-1].bias.detach().clone()
    critic_before = copy.deepcopy(algo.networks.terms["critic"].state_dict())
    rollout(algo, env)
    metrics = algo.update()
    assert metrics["weight/ppo"] == 0.0 and metrics["weight/lower"] == 2.0
    # Action 3 has no teacher: its output bias must not learn when the PPO policy term is zero.
    torch.testing.assert_close(actor.mlp[-1].bias[3], actor_before[3], atol=0, rtol=0)
    assert not torch.equal(actor.mlp[-1].bias[:3], actor_before[:3])
    for key, value in algo.networks.terms["critic"].state_dict().items():
        torch.testing.assert_close(value, critic_before[key], atol=0, rtol=0)
    rollout(algo, env)
    metrics = algo.update()
    assert metrics["weight/ppo"] == 0.5 and metrics["weight/lower"] == 1.0 and metrics["weight/upper"] == 0.5
    assert not torch.equal(actor.mlp[-1].bias[3], actor_before[3])
    assert any(not torch.equal(v, critic_before[k]) for k, v in algo.networks.terms["critic"].state_dict().items())
    checkpoint = algo.save()
    checkpoint["iter"] = 1
    resumed = CompositionalPPO.construct_algorithm(obs, env, cfg, "cpu")
    resumed.load(checkpoint, None, strict=True)
    assert resumed.loss_schedule_step == 8 and resumed.ppo_loss_weight == 1.0
    # A weights-only transfer starts the new schedule at zero.
    transfer = CompositionalPPO.construct_algorithm(obs, env, cfg, "cpu")
    transfer.load(checkpoint, {"networks": True, "iteration": False}, strict=True)
    assert transfer.loss_schedule_step == 0 and transfer.ppo_loss_weight == 0.0

    def reject_teacher_evaluation(module, args):
        raise AssertionError("Zero-weight teachers must not be evaluated")

    for name in ("lower_teacher", "upper_teacher"):
        resumed.networks.terms[name].register_forward_pre_hook(reject_teacher_evaluation)
    rollout(resumed, env)
    metrics = resumed.update()
    assert metrics["weight/ppo"] == 1.0 and metrics["weight/lower"] == metrics["weight/upper"] == 0.0
    assert "loss/lower" not in metrics and "loss/upper" not in metrics


@pytest.mark.parametrize("target,end_value", [("loss.missing.weight", 1.0), ("ppo_loss_weight", -1.0)])
def test_loss_schedule_rejects_unknown_targets_and_negative_weights(
    tmp_path: Path, target: str, end_value: float
) -> None:
    env = DistillationEnvironment()
    obs = env.get_observations()
    cfg = configuration(tmp_path, obs)
    cfg["algorithm"]["curriculum_cfg"] = {
        target: {
            "type": "linear",
            "params": {"start_step": 0, "end_step": 8, "start_value": 1.0, "end_value": end_value},
        }
    }
    with pytest.raises(ValueError, match="Unknown PPO curriculum target|nonnegative weights"):
        CompositionalPPO.construct_algorithm(obs, env, cfg, "cpu")
