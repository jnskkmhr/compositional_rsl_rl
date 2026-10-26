# Compositional FlashSAC

Install directly from GitHub:

```bash
python -m pip install 'git+https://github.com/jnskkmhr/compositional_rsl_rl.git@jnskkmhr/flash-sac'
```

This development version (`0.2.0.dev0`) declares its RSL-RL dependency as a standard direct Git
requirement pinned to `ace7c23aa71a044bd30b0f2d39e8707a87243023` (`5.5.1+flashsac.1`). pip and uv install it automatically,
including when this package is installed as another project's dependency. No local sibling
checkout or dependency-level uv override is required.

IsaacLab pins the compositional Git revision and records both revisions in `uv.lock`. Running
`uv sync --locked` from a fresh IsaacLab checkout installs the pair. When changing either pin,
update the corresponding package metadata and regenerate the consuming project's lockfile.

`CompositionalFlashSAC` subclasses native `FlashSAC`. It supplies network graphs and auxiliary
loss hooks; replay, collection, categorical projection and the SAC update loop live in RSL-RL.
PPO's public configuration and runner remain available. Reusable terms now live in
`compositional_rsl_rl.components`; the previous `mdp` imports and `algorithm.algorithm`
PPO import remain forwarding aliases for existing saved configurations.

## Architecture and configuration

The actor and critic use the existing named-network graph, just like a map encoder feeding
PPO heads. `ResidualNetworkCfg` defines an ordinary feature term under `components/network`;
its `output_dim` is the feature width. Heads reference that term through `input` and select
output distributions with the existing `distribution_cfg` mechanism.

```python
from dataclasses import dataclass
from compositional_rsl_rl.components.network import ResidualNetworkCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg, NetworkCfg

@dataclass(kw_only=True)
class NetworksCfg(NetworkCfg):
    actor_encoder: ResidualNetworkCfg
    critic_encoder: ResidualNetworkCfg

networks = NetworksCfg(
    actor_encoder=ResidualNetworkCfg(
        input=["policy", "command"], output_dim=128, num_blocks=2,
    ),
    critic_encoder=ResidualNetworkCfg(
        input=["critic", "command", "privileged"],
        output_dim=256, num_blocks=2, num_ensembles=2,
    ),
    actor=ActorHeadCfg(
        class_name="compositional_rsl_rl.components.network:LinearNetwork",
        input=["actor_encoder"], hidden_dims=[],
        distribution_cfg=ActorHeadCfg.SquashedGaussianDistributionCfg(
            log_std_min=-10.0, log_std_max=2.0,
        ),
    ),
    critic=CriticHeadCfg(
        class_name="compositional_rsl_rl.components.network:LinearNetwork",
        input=["critic_encoder"], hidden_dims=[],
        distribution_cfg=CriticHeadCfg.CategoricalValueDistributionCfg(
            num_bins=101, value_min=-5.0, value_max=5.0,
        ),
    ),
)
# On CompositionalFlashSACAlgorithmCfg:
# critic_action_input="critic_encoder"
```

`LinearNetwork` contains only normalized output projections; it has no backbone or residual
blocks. It preserves the reference's normalized output weights and independent ensemble
readouts. The actor projects features to mean/log-std; each Q readout projects to return logits.
`num_ensembles=2` keeps both critic feature extractors and their readouts independent. Actor and
critic parameters are also separate. The new distributions work on an ordinary `MLPModel` too;
selecting a different projection architecture does not change distribution configuration.

`critic_action_input` names the existing critic term that receives actions. It defaults to the
critic head; set it to `critic_encoder` to concatenate observations and actions **before** the
residual network. The adapter obtains the additional width from `env.num_actions`; actions are
not environment observation groups. Targets shared with the actor or detached by a separate
optimizer are rejected. The critic support must match the algorithm's bins and symmetric range.

No separate actor/critic wrapper is used. Generic distribution implementations live in the
existing RSL-RL `modules/distribution.py`. The existing `MLPModel` accepts an optional runtime
input through `extra_input_dim`/`extra_input`, and `MLP` accepts `hidden_dims=[]` for a linear
readout. Residual layers, normalization constraints and ensemble projections remain in the
compositional component module. No network is silently constructed when `networks` is missing.

Intermediate terms use the existing network input/dependency interface. The critic head
receives replay actions for critic updates and sampled actions for actor updates. Its target
is a separate copy of the entire critic dependency graph.

Auxiliary terms use `optimizer="actor"`, `"critic"`, or `"separate"`; PPO retains `"joint"`.
Specify `trains` explicitly, as for PPO. Labels come from the sampled replay transition:
`target="label"` reads the current observation group, and `target="next_label"` reads its
actual successor, including the pre-reset observation at a boundary. Raw groups whose names
collide with generated `next_` names are rejected.

Trainable actor and critic graphs must be disjoint. Shared frozen terms are supported.
Ambiguous optimizer ownership, frozen SAC heads, recurrent terms and intermediate empirical
observation normalizers are rejected. The reference components retain the original normalization.
Graph configuration and ownership are checked when restoring native learner checkpoints.

## Validation and integration scope

The reference comparison is in `tests/reference/flash_sac_parity.py` and requires the pinned
`flash_rl` source only for that test. Regular tests require no reference package. They cover
core/composed update equivalence, optimizer ownership, replay-aligned labels, learner restoration,
and runner collection/inference/JIT/ONNX with simulated auto-reset transitions.

The main `/home/jkamohara/isaac/IsaacLab` checkout now selects `OffPolicyRunner` in train/play/export
and preserves final observations with correct history. Rigid G1 rough/flat configs in
`contrib/velocity/config/g1_29dof_rigid/agents/compositional_flash_sac_cfg.py` register
`rsl_rl_flash_sac_cfg_entry_point`. PPO remains the default. LEAPP determines required observation
groups from the constructed policy, including compositional command inputs.

`CompositionalFlashSACAlgorithmCfg.symmetry_cfg` uses the core symmetry extension. Replay augmentation
also transforms auxiliary labels in current and successor observation groups. Optional mirror loss
uses the actor graph and follows the actor optimizer schedule. The G1 example explicitly spells out
Agile learner hyperparameters and enables paired augmentation with mirror loss disabled, matching
Agile's default. Its replay capacity is reduced for this task's larger ten-frame observations.
The G1 example now uses the generic curriculum below for Agile's 1-to-4 action-bound schedule.

The native runner restores learner state with fresh replay, not an exact simulator continuation.
Agile checkpoint conversion and its task-specific relabeling/curricula are not included.
The earlier residual-specific actor/critic classes and configs were removed. Their parameters
now belong to named encoder and head terms in the graph. Earlier development
checkpoints need explicit configuration and state-key migration; automatic migration is not provided.

The named-feature implementation passed 219 core tests and the 100-test composition suite;
an additional checkpoint/action-routing regression also passed with the focused FlashSAC tests.
Eight reference updates matched on both CPU and CUDA FP16 AMP, including weights and normalization
buffers. Native registered G1 training, checkpoint playback, JIT/ONNX and LEAPP export passed.
The IsaacLab formatting hooks passed for the edited files; its global changelog hook still reports
pre-existing unrelated fragment problems.

## Generic curriculum config

The parameter name is the mapping key; schedule type and arguments are separate:

```python
from compositional_rsl_rl.algorithm import CompositionalFlashSACAlgorithmCfg, SchedulerCfg

algorithm = CompositionalFlashSACAlgorithmCfg(
    curriculum_cfg={
        "action_support": SchedulerCfg(
            type="linear",
            params={"start_step": 5000, "end_step": 10000, "start_value": 1.0, "end_value": 4.0},
        ),
        "target_tau": SchedulerCfg(
            type="discrete",
            params={"steps": [0, 10000], "values": [0.01, 0.005]},
        ),
    },
)
```

Steps count completed vectorized collection steps, including warmup. The actual G1 example enables
only the action-support entry; the target-tau entry above illustrates discrete scheduling.
Runtime parameters supported by FlashSAC are `action_support`, `learning_rate`, `target_tau`,
`actor_update_period` (discrete integers), and `temperature_target_sigma`. Learning-rate curricula
replace the built-in cosine schedule for the SAC optimizers; separate auxiliary optimizers retain
their own rates. Unsupported names fail explicitly. PPO curriculum integration is not included.
Schedule progress is checkpointed, saved action bounds are used for export, and replay actions
are never retrospectively rescaled. These additions use existing files only.

Curriculum validation passed 230 core tests and 102 compositional tests. A native G1 smoke test
used a shortened ramp, reached action support four, and verified the applied step in its checkpoint.
Playback, JIT/ONNX and LEAPP export from that checkpoint passed. Source lint and edited-file
formatting passed; IsaacLab's global changelog check still reports unrelated existing fragment errors.
