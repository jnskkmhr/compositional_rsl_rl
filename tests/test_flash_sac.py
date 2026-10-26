"""Composition must preserve native FlashSAC updates and explicit gradient ownership."""

import copy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import FlashSAC
from tensordict import TensorDict

from compositional_rsl_rl import CompositionalFlashSAC
from compositional_rsl_rl.components.network import ResidualNetworkCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg


def mirror(env, obs=None, actions=None):
    """Append sign-reflected observation groups, including auxiliary labels."""
    return (
        torch.cat((obs, -obs)) if obs is not None else None,
        torch.cat((actions, -actions)) if actions is not None else None,
    )


def setup():
    obs = TensorDict({"policy": torch.randn(8, 3), "label": torch.randn(8, 2)}, [8])
    cfg = {
        "algorithm": {
            "class_name": "FlashSAC",
            "num_bins": 11,
            "normalize_reward": False,
            "critic_action_input": "critic_encoder",
        },
    }
    cfg["networks"] = {
        "actor_encoder": asdict(ResidualNetworkCfg(input=["policy"], output_dim=8, num_blocks=1)),
        "critic_encoder": asdict(ResidualNetworkCfg(input=["policy"], output_dim=8, num_blocks=1, num_ensembles=2)),
        "actor": asdict(
            ActorHeadCfg(
                class_name="compositional_rsl_rl.components.network:LinearNetwork",
                input=["actor_encoder"],
                hidden_dims=[],
                distribution_cfg=ActorHeadCfg.SquashedGaussianDistributionCfg(),
            )
        ),
        "critic": asdict(
            CriticHeadCfg(
                class_name="compositional_rsl_rl.components.network:LinearNetwork",
                input=["critic_encoder"],
                hidden_dims=[],
                distribution_cfg=CriticHeadCfg.CategoricalValueDistributionCfg(num_bins=11),
            )
        ),
    }
    batch = TensorDict(
        {
            "observations": obs,
            "next_observations": obs.clone(),
            "actions": torch.zeros(8, 1),
            "rewards": torch.randn(8),
            "terminated": torch.zeros(8, dtype=torch.bool),
            "bootstrap_discount": torch.full((8,), 0.99),
        },
        [8],
    )
    return obs, cfg, batch


def test_composition_without_auxiliary_loss_matches_native_updates():
    obs, cfg, batch = setup()
    env = SimpleNamespace(num_actions=1)
    comp = CompositionalFlashSAC.construct_algorithm(obs, env, cfg, "cpu")
    plain = FlashSAC(
        copy.deepcopy(comp.actor),
        copy.deepcopy(comp.critic),
        1,
        comp.obs_groups,
        num_bins=11,
        normalize_reward=False,
    )
    # The native update consumes concatenated observations; these graph views consume TensorDicts.
    plain.actor_observations = lambda obs: obs
    plain.critic_observations = lambda obs: obs
    for step in range(3):
        torch.manual_seed(step)
        expected = plain.update(batch)
        torch.manual_seed(step)
        actual = comp.update(batch)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
        for name in ("actor", "critic", "target_critic"):
            graph = getattr(comp, name)
            for key, value in graph.state_dict().items():
                torch.testing.assert_close(value, getattr(plain, name).state_dict()[key])
    torch.testing.assert_close(comp.get_policy()(obs), plain.deterministic_actions(obs))


@pytest.mark.parametrize("phase", ["actor", "critic", "separate"])
def test_auxiliary_losses_train_only_selected_replay_predictor(phase):
    obs, cfg, batch = setup()
    cfg["algorithm"]["symmetry_cfg"] = {"data_augmentation_func": mirror, "use_data_augmentation": True}
    cfg["networks"]["estimator"] = {"class_name": "MLPModel", "input": ["policy"], "output_dim": 2, "hidden_dims": [8]}
    cfg["loss"] = {
        "estimate": {
            "prediction": "estimator",
            "target": "next_label",
            "trains": ["estimator"],
            "optimizer": phase,
            "learning_rate": 1e-3 if phase == "separate" else None,
        }
    }
    batch["next_observations", "label"] = torch.ones(8, 2) * 3
    alg = CompositionalFlashSAC.construct_algorithm(obs, SimpleNamespace(num_actions=1), cfg, "cpu")
    before = copy.deepcopy(alg.networks.terms["estimator"].state_dict())
    state = copy.deepcopy(alg.save())
    alg.update(batch)
    assert any(
        not torch.equal(value, before[key]) for key, value in alg.networks.terms["estimator"].state_dict().items()
    )
    restored = CompositionalFlashSAC.construct_algorithm(obs, SimpleNamespace(num_actions=1), cfg, "cpu")
    restored.load(state)
    for key, value in restored.networks.terms["estimator"].state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert set(restored.get_policy().actor.terms) == {"actor_encoder", "actor"}


def test_trainable_shared_encoder_is_rejected_and_frozen_encoder_is_supported():
    obs, cfg, batch = setup()
    cfg["networks"]["encoder"] = {"class_name": "MLPModel", "input": ["policy"], "output_dim": 3, "hidden_dims": [8]}
    for head in ("actor", "critic"):
        cfg["networks"][head + "_encoder"]["input"] = ["encoder"]
    with pytest.raises(ValueError, match="share trainable"):
        CompositionalFlashSAC.construct_algorithm(obs, SimpleNamespace(num_actions=1), cfg, "cpu")
    cfg["networks"]["encoder"]["frozen"] = True
    alg = CompositionalFlashSAC.construct_algorithm(obs, SimpleNamespace(num_actions=1), cfg, "cpu")
    before = copy.deepcopy(alg.networks.terms["encoder"].state_dict())
    alg.update(batch)
    for key, value in alg.networks.terms["encoder"].state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_compositional_runner_checkpoint_and_actor_exports(tmp_path):
    """Exercise the moved architecture through collection, restoration, JIT and ONNX."""
    import onnxruntime
    from rsl_rl.runners import OffPolicyRunner
    from compositional_rsl_rl.runner import CompositionalOffPolicyRunnerCfg

    obs, cfg, _ = setup()

    class Env:
        num_actions = 1
        num_envs = 8
        device = "cpu"
        cfg = {}
        max_episode_length = 3
        episode_length_buf = torch.zeros(8, dtype=torch.long)

        def get_observations(self):
            return obs

        def step(self, actions):
            self.episode_length_buf += 1
            done = self.episode_length_buf % 3 == 0
            final = obs.clone() + 1
            return (
                obs,
                -actions.square().sum(-1),
                done,
                {
                    "terminated": torch.zeros_like(done),
                    "truncated": done,
                    "final_obs": final,
                    "final_obs_mask": done,
                },
            )

    runner_cfg = asdict(
        CompositionalOffPolicyRunnerCfg(
            networks=cfg["networks"],
            batch_size=8,
            warmup_transitions=8,
            replay_capacity=64,
            replay_max_bytes=1_000_000,
            updates_per_step=1,
        )
    )
    runner_cfg["algorithm"].update(num_bins=11, normalize_reward=False, critic_action_input="critic_encoder")
    runner = OffPolicyRunner(Env(), runner_cfg)
    runner.learn(6)
    assert runner.alg.update_step == 4
    assert runner.replay.storage[0]["next_observations", "policy"].equal(obs[0]["policy"] + 1)
    runner.save(str(tmp_path / "checkpoint.pt"))
    restored = OffPolicyRunner(Env(), runner_cfg)
    with pytest.warns(UserWarning, match="fresh replay"):
        restored.load(str(tmp_path / "checkpoint.pt"))
    assert restored.replay is None
    expected = restored.get_inference_policy()(obs[:2])
    torch.testing.assert_close(expected, runner.get_inference_policy()(obs[:2]))
    restored.export_policy_to_jit(str(tmp_path))
    torch.testing.assert_close(torch.jit.load(str(tmp_path / "policy.pt"))(obs[:2]["policy"]), expected)
    restored.export_policy_to_onnx(str(tmp_path))
    session = onnxruntime.InferenceSession(str(tmp_path / "policy.onnx"), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"obs": obs[:2]["policy"].numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(actual), expected, atol=1e-5, rtol=1e-5)


def test_q_input_width_and_action_gradient_are_independent_of_backbone():
    """Actions stay outside observation groups and Q remains differentiable with respect to them."""
    obs, cfg, batch = setup()
    for use_residual in (True, False):
        if not use_residual:
            for head in ("actor", "critic"):
                cfg["networks"].pop(head + "_encoder")
                cfg["networks"][head].update(class_name="MLPModel", input=["policy"], hidden_dims=[8])
            cfg["algorithm"]["critic_action_input"] = "critic"
        alg = CompositionalFlashSAC.construct_algorithm(obs, SimpleNamespace(num_actions=1), cfg, "cpu")
        assert alg.obs_groups["critic"] == ["policy"]
        name = "critic_encoder" if use_residual else "critic"
        model = alg.networks.terms[name]
        assert model.obs_dim == 3 and model.extra_input_dim == 1
        if use_residual:
            assert model.mlp.embedder.norm.running_mean.shape == (2, 4)
        else:
            assert model.mlp[0].in_features == 4
        actions = batch["actions"].clone().requires_grad_()
        values, _ = alg.critic(obs, actions, training=False)
        assert values.shape == (2 if use_residual else 1, 8)
        gradient = torch.autograd.grad(values.sum(), actions)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        assert all(torch.isfinite(value) for value in alg.update(batch).values())


def test_distribution_settings_reach_actor_and_critic_projections():
    """Public head parameters control actual scales and return support."""
    obs, cfg, _ = setup()
    cfg["networks"]["actor"]["distribution_cfg"].update(log_std_min=-3.0, log_std_max=-1.0)
    cfg["networks"]["critic"]["distribution_cfg"].update(value_min=-7.0, value_max=7.0)
    cfg["algorithm"]["value_max"] = 7.0
    alg = CompositionalFlashSAC.construct_algorithm(obs, SimpleNamespace(num_actions=1), cfg, "cpu")
    _, std = alg.actor.get_mean_and_std(obs, training=False)
    assert (std >= torch.exp(torch.tensor(-3.0))).all()
    assert (std <= torch.exp(torch.tensor(-1.0))).all()
    torch.testing.assert_close(alg.critic.value_support, torch.linspace(-7, 7, 11))
    cfg["networks"]["actor"]["distribution_cfg"].update(log_std_min=2.0, log_std_max=-1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        CompositionalFlashSAC.construct_algorithm(obs, SimpleNamespace(num_actions=1), cfg, "cpu")


def test_residual_backbone_with_existing_ppo_heads_and_gaussian_settings():
    """The same feature extractor also feeds a Gaussian PPO actor and scalar state-value critic."""
    from compositional_rsl_rl.network import NetworkManager, GraphModel

    obs, cfg, _ = setup()
    cfg["networks"]["actor"]["distribution_cfg"] = asdict(ActorHeadCfg.GaussianDistributionCfg(init_std=0.7))
    for head in ("actor", "critic"):
        cfg["networks"][head]["class_name"] = "MLPModel"
    cfg["networks"]["critic"]["distribution_cfg"] = None
    cfg["networks"]["critic_encoder"]["num_ensembles"] = 1
    manager = NetworkManager(obs, cfg["networks"], num_actions=1)
    actor = GraphModel(manager, "actor")
    critic = GraphModel(manager, "critic")
    actions = actor(obs, stochastic_output=True)
    assert actions.shape == (8, 1)
    torch.testing.assert_close(actor.output_std, torch.full((8, 1), 0.7))
    assert torch.isfinite(actor.get_output_log_prob(actions)).all()
    assert critic(obs).shape == (8, 1)
    actor.eval()
    expected = actor(obs)
    exported = torch.jit.script(actor.as_jit())
    torch.testing.assert_close(exported(obs["policy"]), expected)


def test_action_routing_is_validated_and_preserved_in_checkpoints():
    """Actor branches cannot consume critic actions, and resumes cannot silently change routing."""
    obs, cfg, _ = setup()
    env = SimpleNamespace(num_actions=1)
    alg = CompositionalFlashSAC.construct_algorithm(obs, env, cfg, "cpu")
    state = alg.save()
    assert state["critic_action_input"] == "critic_encoder"
    state["critic_action_input"] = "critic"
    with pytest.raises(ValueError, match="routing differs"):
        alg.load(state)
    cfg["algorithm"]["critic_action_input"] = "actor_encoder"
    with pytest.raises(ValueError, match="exclusively to the critic"):
        CompositionalFlashSAC.construct_algorithm(obs, env, cfg, "cpu")


def test_curriculum_cfg_serialization_and_compositional_optimizer_replacement():
    """Dataclass schedule entries drive the actual composed optimizer groups and policy scale."""
    from compositional_rsl_rl.algorithm import SchedulerCfg

    obs, cfg, batch = setup()
    cfg["algorithm"]["curriculum_cfg"] = {
        "learning_rate": asdict(SchedulerCfg(type="discrete", params={"steps": [0, 2], "values": [1e-4, 5e-5]})),
        "action_support": asdict(
            SchedulerCfg(
                type="linear",
                params={
                    "start_step": 0,
                    "end_step": 2,
                    "start_value": 1.0,
                    "end_value": 3.0,
                },
            )
        ),
    }
    alg = CompositionalFlashSAC.construct_algorithm(obs, SimpleNamespace(num_actions=1), cfg, "cpu")
    assert all(opt.param_groups[0]["lr"] == pytest.approx(1e-4) for opt in alg.optimizers.values())
    alg.update_curriculum(2)
    alg.update(batch)
    assert alg.get_policy().action_support == 3
    assert all(opt.param_groups[0]["lr"] == pytest.approx(5e-5) for opt in alg.optimizers.values())
    restored = CompositionalFlashSAC.construct_algorithm(obs, SimpleNamespace(num_actions=1), cfg, "cpu")
    restored.load(copy.deepcopy(alg.save()))
    assert restored.curriculum_step == 2 and restored.action_support == 3
    torch.testing.assert_close(restored.get_policy()(obs), alg.get_policy()(obs))
