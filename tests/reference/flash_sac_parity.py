# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare native learner updates with the pinned FlashSAC source on CPU or CUDA."""

import argparse
import numpy as np
import subprocess
import torch
from pathlib import Path
from tensordict import TensorDict
from types import SimpleNamespace

import flash_rl
import gymnasium as gym
from flash_rl.agents.flashSAC.agent import FlashSACAgent, FlashSACConfig

from compositional_rsl_rl import CompositionalFlashSAC
from compositional_rsl_rl.components.network import ResidualNetworkCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg
from dataclasses import asdict

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", default="cpu")
parser.add_argument("--amp", action="store_true")
args = parser.parse_args()
reference_root = Path(flash_rl.__file__).resolve().parent.parent
revision = subprocess.check_output(["git", "-C", str(reference_root), "rev-parse", "HEAD"], text=True).strip()
assert revision == "11a276cad45ba8816639f66c210dfc2db755bb2c", revision
torch.set_default_device(args.device)
torch.set_num_threads(1)
config = FlashSACConfig(
    seed=1,
    normalize_reward=False,
    normalized_G_max=5.0,
    asymmetric_observation=True,
    device_type=args.device,
    buffer_max_length=1,
    buffer_min_length=1,
    buffer_device_type=args.device,
    sample_batch_size=8,
    learning_rate_init=3e-4,
    learning_rate_peak=3e-4,
    learning_rate_end=1.5e-4,
    learning_rate_warmup_rate=0.0,
    learning_rate_warmup_step=0,
    learning_rate_decay_rate=1.0,
    learning_rate_decay_step=100000,
    actor_num_blocks=1,
    actor_hidden_dim=8,
    actor_bc_alpha=0.0,
    actor_noise_zeta_mu=2.0,
    actor_noise_zeta_max=16,
    actor_update_period=2,
    critic_num_blocks=1,
    critic_hidden_dim=8,
    critic_num_bins=11,
    critic_min_v=-5.0,
    critic_max_v=5.0,
    critic_target_update_tau=0.01,
    temp_initial_value=0.01,
    temp_target_sigma=0.15,
    temp_target_entropy=0.0,
    gamma=0.99,
    n_step=1,
    use_compile=False,
    compile_mode="default",
    use_amp=args.amp,
    load_optimizer=True,
    load_reward_normalizer=True,
)
torch.manual_seed(23)
ref = FlashSACAgent(
    gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
    gym.spaces.Box(-1, 1, (2,), np.float32),
    {"actor_observation_size": (3,)},
    config,
    use_internal_replay=False,
)
obs = TensorDict({"policy": torch.randn(8, 3), "critic": torch.randn(8, 4)}, [8])
cfg = {
    "algorithm": {
        "class_name": "FlashSAC",
        "num_bins": 11,
        "normalize_reward": False,
        "use_amp": args.amp,
        "critic_action_input": "critic_encoder",
    }
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
cfg["networks"]["critic_encoder"]["input"] = ["critic"]
alg = CompositionalFlashSAC.construct_algorithm(obs, SimpleNamespace(num_actions=2), cfg, args.device)


def reference_state(model):
    """Map graph term state to the reference layout without changing any numerical values."""
    if not hasattr(model, "terms"):
        return model.state_dict()
    state = {name: value for name, value in model.terms[model.head + "_encoder"].mlp.state_dict().items()}
    head = model.terms[model.head]
    projection = head.mlp
    if model.head == "actor":
        for i, name in enumerate(("mean", "std")):
            state["predictor." + name + "_w.w.weight"] = projection.layers[i].w.weight
            state["predictor." + name + "_bias"] = projection.biases[i]
    else:
        state["predictor.w.weight"] = projection.layers[0].weight
        state["predictor.bias"] = projection.biases[0]
        state["predictor.bin_values"] = head.distribution.bin_values.reshape(1, 1, -1)
    return state


models = alg.models
with torch.no_grad():
    for key, model in models.items():
        for name, tensor in reference_state(model).items():
            tensor.copy_(getattr(ref, "_" + key).network.state_dict()[name])
batch = TensorDict(
    {
        "observations": obs,
        "next_observations": TensorDict({"policy": torch.randn(8, 3), "critic": torch.randn(8, 4)}, [8]),
        "actions": torch.rand(8, 2) * 2 - 1,
        "rewards": torch.randn(8),
        "terminated": torch.tensor([True, False] * 4),
        "bootstrap_discount": torch.tensor([0.99, 0.99**2, 0.99**3, 0.99] * 2),
    },
    [8],
)
b = {
    "actor_observation": obs["policy"],
    "observation": obs["critic"],
    "actor_next_observation": batch["next_observations", "policy"],
    "next_observation": batch["next_observations", "critic"],
    "action": batch["actions"],
    "reward": batch["rewards"],
    "terminated": batch["terminated"].float(),
    "bootstrap_discount": batch["bootstrap_discount"],
}
for i in range(8):
    torch.manual_seed(i + 10)
    expected = ref.update_from_batch(b)
    torch.manual_seed(i + 10)
    actual = alg.update(batch)
    for key in expected:
        torch.testing.assert_close(
            actual[key], expected[key], rtol=2e-5, atol=2e-6, msg=lambda s, key=key: key + " " + s
        )
    for key, model in models.items():
        for name, value in reference_state(model).items():
            torch.testing.assert_close(
                value,
                getattr(ref, "_" + key).network.state_dict()[name],
                rtol=2e-5,
                atol=2e-6,
                msg=lambda s, key=key, name=name: key + "." + name + " " + s,
            )
print("PASS: 8 updates match pinned reference metrics and model state, including shortened n-step discounts.")
