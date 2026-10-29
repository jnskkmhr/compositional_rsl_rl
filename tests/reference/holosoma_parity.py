"""Compare actual holosoma_agent update methods against compositional FastSAC.

Run with --reference-root pointing to jnskkmhr/holosoma_agent at a7fe4c9.
The AST loader excludes logging/environment dependencies, not learning computations.
"""

import argparse
import ast
import copy
import importlib.util
import subprocess
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from compositional_rsl_rl import CompositionalFastSAC
from compositional_rsl_rl.algorithm import CompositionalFastSACAlgorithmCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference-root", type=Path, required=True)
parser.add_argument("--normalize-obs", action="store_true")
args = parser.parse_args()
revision = subprocess.check_output(["git", "-C", str(args.reference_root), "rev-parse", "HEAD"], text=True).strip()
assert revision == "a7fe4c941fb52fb71c70158ac000968b01f32d1b", revision
source = args.reference_root / "holosoma_agent/algorithms/fast_sac"
spec = importlib.util.spec_from_file_location("reference_networks", source / "networks.py")
networks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(networks)
tree = ast.parse((source / "fast_sac.py").read_text())
agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FastSACAgent")
agent.bases = []
agent.body = [
    node
    for node in agent.body
    if isinstance(node, ast.FunctionDef) and node.name in ("_update_critic", "_update_policy")
]
scope = {"torch": torch, "TensorDict": TensorDict, "F": F, "nn": torch.nn}
exec(
    compile(ast.Module(body=[agent], type_ignores=[]), str(source / "fast_sac.py"), "exec"),
    scope,
)
normalizer_tree = ast.parse((source / "fast_sac_utils.py").read_text())
normalizer_class = next(
    node for node in normalizer_tree.body if isinstance(node, ast.ClassDef) and node.name == "EmpiricalNormalization"
)
# Correct the reference's pooled-variance bug: the between-batch correction
# uses the OLD mean difference (delta), not the post-update delta2.
corrections = 0
for node in ast.walk(normalizer_class):
    if (
        (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "pow")
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "delta2"
    ):
        node.func.value.id = "delta"
        corrections += 1
assert corrections == 1
scope["dist"] = torch.distributed
exec(
    compile(
        ast.Module(body=[normalizer_class], type_ignores=[]),
        str(source / "fast_sac_utils.py"),
        "exec",
    ),
    scope,
)
ref_actor_norm = scope["EmpiricalNormalization"](3, "cpu")
ref_critic_norm = scope["EmpiricalNormalization"](4, "cpu")
ref = scope["FastSACAgent"]()
ref.config = SimpleNamespace(gamma=0.97, max_grad_norm=0.0, use_autotune=True)
ref.device = torch.device("cpu")
ref.is_multi_gpu = False
ref.amp_context = nullcontext
ref.scaler = torch.amp.GradScaler("cpu", enabled=False)
ref.target_entropy = 0.0
torch.set_num_threads(1)
torch.manual_seed(23)
ref.actor = networks.Actor(3, 2, [8, 6], 0.0, -5.0, True, True, device="cpu", action_scale=torch.ones(2))
ref.qnet = networks.Critic(4, 2, [8, 6], 11, -20.0, 20.0, True, 2, torch.device("cpu"))
ref.qnet_target = copy.deepcopy(ref.qnet)
ref.log_alpha = torch.tensor([0.001]).log().requires_grad_()
ref.actor_optimizer = torch.optim.AdamW(ref.actor.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.001)
ref.q_optimizer = torch.optim.AdamW(ref.qnet.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.001)
ref.alpha_optimizer = torch.optim.AdamW([ref.log_alpha], lr=3e-4, betas=(0.9, 0.95))
obs = TensorDict({"policy": torch.randn(8, 3), "critic": torch.randn(8, 4)}, [8])
cfg = {
    "algorithm": asdict(CompositionalFastSACAlgorithmCfg(num_bins=11, obs_normalization=args.normalize_obs)),
    "networks": {
        "actor": asdict(
            ActorHeadCfg(
                input=["policy"],
                hidden_dims=[8, 6],
                activation="swish",
                layer_norm=True,
                distribution_cfg=ActorHeadCfg.SquashedGaussianDistributionCfg(log_std_min=-5.0, log_std_max=0.0),
            )
        ),
        "critic": asdict(
            CriticHeadCfg(
                input=["critic"],
                hidden_dims=[8, 6],
                activation="swish",
                layer_norm=True,
                num_ensembles=2,
                distribution_cfg=CriticHeadCfg.CategoricalValueDistributionCfg(
                    num_bins=11, value_min=-20.0, value_max=20.0
                ),
            )
        ),
    },
}
alg = CompositionalFastSAC.construct_algorithm(obs, SimpleNamespace(num_actions=2), cfg, "cpu")
actor = alg.networks.terms["actor"].mlp
with torch.no_grad():
    for actual, expected in zip(list(actor)[:6], ref.actor.net):
        actual.load_state_dict(expected.state_dict())
    actor[6].weight.copy_(torch.cat([ref.actor.fc_mean.weight, ref.actor.fc_log_std.weight]))
    actor[6].bias.copy_(torch.cat([ref.actor.fc_mean.bias, ref.actor.fc_log_std.bias]))
    for actual, expected in zip(alg.networks.terms["critic"].mlp.members, ref.qnet.qnets):
        actual.load_state_dict(expected.net.state_dict())
    alg.target_critic.load_state_dict(alg.critic.state_dict())
for step in range(8):
    torch.manual_seed(100 + step)
    nxt = TensorDict({"policy": torch.randn(8, 3), "critic": torch.randn(8, 4)}, [8])
    terminal = torch.tensor([False, True, False, True, False, False, False, False])
    timeout = torch.tensor([False, False, True, False, False, False, False, False])
    horizon = torch.tensor([1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0])
    batch = TensorDict(
        {
            "observations": obs,
            "next_observations": nxt,
            "actions": torch.randn(8, 2),
            "rewards": torch.randn(8),
            "terminated": terminal,
            "bootstrap_discount": 0.97**horizon,
        },
        [8],
    )
    reference_batch = TensorDict(
        {
            "observations": obs["policy"],
            "critic_observations": obs["critic"],
            "actions": batch["actions"],
            "next": {
                "observations": nxt["policy"],
                "critic_observations": nxt["critic"],
                "rewards": batch["rewards"],
                "dones": terminal | timeout,
                "truncations": timeout,
                "effective_n_steps": horizon,
            },
        },
        [8],
    )
    if args.normalize_obs:
        reference_batch["observations"] = ref_actor_norm(reference_batch["observations"])
        reference_batch["next", "observations"] = ref_actor_norm(reference_batch["next", "observations"])
        reference_batch["critic_observations"] = ref_critic_norm(reference_batch["critic_observations"])
        reference_batch["next", "critic_observations"] = ref_critic_norm(reference_batch["next", "critic_observations"])
    torch.manual_seed(step)
    metrics = alg.update(batch)
    torch.manual_seed(step)
    result = ref._update_critic(reference_batch)
    torch.testing.assert_close(metrics["critic/loss"], result[2], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(metrics["temperature/loss"], result[5], rtol=2e-5, atol=2e-6)
    if step % 4 == 0:
        result = ref._update_policy(reference_batch)
        torch.testing.assert_close(metrics["actor/loss"], result[1], rtol=2e-5, atol=2e-6)
    with torch.no_grad():
        for target, online in zip(ref.qnet_target.parameters(), ref.qnet.parameters()):
            target.lerp_(online, 0.125)
    for actual, expected in zip(alg.networks.terms["critic"].mlp.members, ref.qnet.qnets):
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, expected.net.state_dict()[key], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        actor[6].weight,
        torch.cat([ref.actor.fc_mean.weight, ref.actor.fc_log_std.weight]),
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(alg.temperature.log_temp, ref.log_alpha, rtol=2e-5, atol=2e-6)
print(
    "PASS: 8 updates agree with holosoma_agent critic/actor/temperature losses and parameters."
    + (" Reference variance correction applied." if args.normalize_obs else "")
)
