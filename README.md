# Compositional RL

An independently installable extension to the **RSL-RL FlashSAC fork (`5.5.1+flashsac.1`)**. Network terms and loss terms are
configured separately. Actor, critic, estimator, and decoder are peer terms; there is no special encoder
container and no mandatory MLP appended to a separate composition model.

Start with the [complete IsaacLab config](#complete-isaaclab-agent-configuration), then see
[architecture swaps](#swapping-model-architectures-in-isaaclab) and
[loss swaps](#swapping-loss-types-and-functions-in-isaaclab).

The GitHub repository, Python package, and installable distribution are named
`compositional_rsl_rl`.

## Installation and environment integration

Install the FlashSAC branch directly into your training environment:

```bash
python -m pip install 'git+https://github.com/jnskkmhr/compositional_rsl_rl.git@jnskkmhr/flash-sac'
```

This automatically installs the compatible RSL-RL fork from the exact Git commit declared in
`project.dependencies`. No sibling checkout or separate RSL-RL install is required.
IsaacLab pins this package to a commit in its uv configuration and lockfile; `uv sync` fetches
both libraries. IsaacLab does not need its own RSL-RL version or source override.

For development, clone this branch and run `python -m pip install -e .`; its RSL-RL dependency
is still installed automatically. PPO uses `OnPolicyRunner`; FlashSAC uses `OffPolicyRunner`.
See [FlashSAC configuration](docs/flash_sac.md).
The runtime and all configs import no simulator. Configs are standard Python dataclasses;
serialize them with `dataclasses.asdict()`, or supply plain dictionaries using the same schema.
An environment must implement the RSL-RL `VecEnv` protocol, usually through its existing adapter.
The optional terrain checkpoint interface has been exercised with Isaac Lab's Newton moving-patch MPM task.
mjlab has not been integration-tested here.

For PPO, import configs from `compositional_rsl_rl.network`, `compositional_rsl_rl.loss`, `compositional_rsl_rl.algorithm`,
and `compositional_rsl_rl.runner`. Keep the runner's `class_name="OnPolicyRunner"`; the algorithm config
selects `class_name="compositional_rsl_rl:CompositionalPPO"`. Ordinary RSL-RL configs and checkpoints
continue to use upstream PPO unchanged.

For PPO in IsaacLab, select your task config through its existing `rsl_rl_cfg_entry_point` and retain the
`rsl_rl` backend. Its train/play/export scripts use the upstream runner and need no changes for this
extension. Only its task example uses IsaacLab's `configclass` decorator to support its CLI utilities.

See [the complete IsaacLab example](examples/isaaclab_g1.py). Its estimator expects a raw observation group
`privileged_state` of shape `[num_envs, 3]`, aligned with the current policy observations. Supply that group
in the environment or adapt the target name and estimator output dimension. Labels are not automatically
sliced from `privileged`, and next-observation targets are not inferred.

## What `@dataclass(kw_only=True)` means

`dataclass` is a Python standard-library decorator. It generates a constructor and other methods
from annotated fields. `kw_only=True` makes constructor arguments keyword-only, so you write
`PointNetNetworkCfg(input=["points"], output_dim=32)` instead of relying on positional argument order.
It does not change the network computation. Use `dataclasses.asdict(cfg)` to serialize these configs
for RSL-RL outside IsaacLab.

Inside an IsaacLab task, you can use its familiar `@configclass` decorator on subclasses of the
library configs, as below. It adds IsaacLab's configuration utilities and handles mutable defaults.
The library itself uses standard dataclasses so importing it never requires a simulator.

## Complete IsaacLab agent configuration

Save the following as `agents/compositional_ppo_cfg.py` in your task package. This is a complete
agent configuration for the existing RSL-RL training backend; your task still supplies the robot,
scene, observations, rewards, and termination conditions. The same executable example is available
in [examples/isaaclab_g1.py](examples/isaaclab_g1.py).

The environment must return these observation groups. `B` is the number of environments; the
vector widths are inferred from the returned tensors.

| Group | Shape | Purpose |
| --- | --- | --- |
| `policy` | `[B, P]` | Actor and estimator inputs. |
| `command` | `[B, C]` | Actor and critic inputs. |
| `critic` | `[B, Q]` | Critic inputs. |
| `privileged` | `[B, R]` | Additional critic inputs. |
| `privileged_state` | `[B, 3]` | Three task-defined supervision values aligned with the current observation. |

`privileged_state` is a separate observation group; it is not automatically sliced from `privileged`.
Provide its labels in your environment observation config, or change the target name and estimator
output dimension to match an existing group. The actor does not consume these labels at inference.

```python
# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Full IsaacLab task config. The environment must return privileged_state with shape [num_envs, 3]."""

from compositional_rsl_rl.algorithm import CompositionalPPOAlgorithmCfg
from compositional_rsl_rl.loss import LossesCfg, SupervisedLossCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg, MLPNetworkCfg, NetworkCfg
from compositional_rsl_rl.runner import CompositionalRunnerCfg
from isaaclab.utils.configclass import configclass


@configclass
class MyNetworksCfg(NetworkCfg):
    """Actor and critic are peers of the learned state estimator."""

    state_estimator: MLPNetworkCfg | None = None


@configclass
class MyLossesCfg(LossesCfg):
    """Named loss terms are independent of network construction."""

    estimator_loss: SupervisedLossCfg | None = None


@configclass
class MyCompositionalPPORunnerCfg(CompositionalRunnerCfg):
    """Select through the task's existing rsl_rl_cfg_entry_point; keep train/play scripts unchanged."""

    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 500

    # Raw observation dependencies are inferred from each network term's input list.
    networks = MyNetworksCfg(
        actor=ActorHeadCfg(
            input=["policy", "state_estimator", "command"],
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=ActorHeadCfg.GaussianDistributionCfg(init_std=1.0),
        ),
        critic=CriticHeadCfg(
            input=["critic", "privileged", "command"],
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
        ),
        state_estimator=MLPNetworkCfg(
            input=["policy"],
            output_dim=3,
            hidden_dims=[256, 128],
            activation="elu",
            obs_normalization=False,
            frozen=False,
            pretrained_cfg=None,
        ),
    )

    loss = MyLossesCfg(
        estimator_loss=SupervisedLossCfg(
            trains=["state_estimator"],
            prediction="state_estimator",
            target="privileged_state",
            loss_fn="mse",
            optimizer="joint",
            weight=1.0,
            # For separate supervision: optimizer="separate", learning_rate=1.0e-3.
        ),
    )

    algorithm = CompositionalPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Enable your RslRlSymmetryCfg only when its observation transform handles all supplied groups.
        # Upstream RSL-RL disallows symmetry for recurrent actor/critic models.
        symmetry_cfg=None,
    )

    logger = "wandb"
    wandb_project = "g1_29dof_rigid_rough_ppo"
    experiment_name = "g1_29dof_rigid_rough_compositional_ppo"
```

In the task's existing `gym.register(..., kwargs={...})`, set its agent entry to the import path of
the class above. For a task package named `my_project.tasks.g1`, the relevant entry is:

```python
{
    "rsl_rl_cfg_entry_point": ("my_project.tasks.g1.agents.compositional_ppo_cfg:MyCompositionalPPORunnerCfg"),
}
```

Replace that package path with your actual task package and retain the other registration entries.
Install `compositional_rsl_rl` in the same Python environment as IsaacLab, then use the task's existing
RSL-RL train/play scripts. The config retains `class_name="OnPolicyRunner"` and selects
`algorithm.class_name="compositional_rsl_rl:CompositionalPPO"`; no script edits are needed for the
supported RSL-RL 5.5.1 interface.

## Swapping model architectures in IsaacLab

Place the following variants in the same task agent file, below the complete base config. Each
variant starts from a fresh base config. Select a variant by changing the class name in the task's
`rsl_rl_cfg_entry_point`. To combine changes, derive a variant from the chosen architecture config
and copy its `networks` or `loss` defaults rather than the original base defaults.

### Change MLP sizes

```python
@configclass
class MySmallMLPPPORunnerCfg(MyCompositionalPPORunnerCfg):
    networks = MyCompositionalPPORunnerCfg().networks
    networks.actor.hidden_dims = [256, 128]
    networks.critic.hidden_dims = [256, 128]
    networks.state_estimator.hidden_dims = [128, 64]
```

### Replace the estimator with PointNet

The environment must additionally provide `points` with shape `[B, N, 4]`: XYZ in the first three
channels and validity in the fourth. This group must preserve its point and channel dimensions.
The term still outputs three values, so the actor references and supervision target stay compatible.

```python
from compositional_rsl_rl.components.network import PointNetNetworkCfg


@configclass
class MyPointNetPPORunnerCfg(MyCompositionalPPORunnerCfg):
    networks = MyCompositionalPPORunnerCfg().networks
    networks.state_estimator = PointNetNetworkCfg(
        input=["points"],
        output_dim=3,
        feature_dim=128,
        pointnet_hidden_dims=[64, 128],
        hidden_dims=[128],
        obs_normalization=False,
    )
```

PointNet applies a shared MLP to valid points and masked max pooling. The actor still receives
`policy`, `state_estimator`, and `command`; `state_estimator` now derives its output from `points`.

### Replace the estimator with a Transformer

This variant uses the same point-cloud observation and validity flag. Positional embeddings are
disabled because this example treats points as an unordered set.

```python
from compositional_rsl_rl.components.network import TransformerNetworkCfg


@configclass
class MyTransformerPPORunnerCfg(MyCompositionalPPORunnerCfg):
    networks = MyCompositionalPPORunnerCfg().networks
    networks.state_estimator = TransformerNetworkCfg(
        input=["points"],
        output_dim=3,
        feature_dim=128,
        num_heads=4,
        num_layers=2,
        feedforward_dim=256,
        hidden_dims=[128],
        validity_channel=True,
        use_positional_encoding=False,
        obs_normalization=False,
    )
```

For ordered tokens, use `input=["tokens"]` with a `[B, N, D]` observation and enable positional
embeddings. Set `validity_channel=False` if every token is valid and all channels are data.
`feature_dim` must be divisible by `num_heads`. This is attention within each observation; it does
not maintain hidden state across environment steps.

### Replace the actor and critic with recurrent heads

```python
from compositional_rsl_rl.network import RecurrentActorHeadCfg, RecurrentCriticHeadCfg


@configclass
class MyRecurrentPPORunnerCfg(MyCompositionalPPORunnerCfg):
    networks = MyCompositionalPPORunnerCfg().networks
    networks.actor = RecurrentActorHeadCfg(
        input=["policy", "state_estimator", "command"],
        hidden_dims=[256, 128],
        rnn_type="lstm",  # Use "gru" for a GRU.
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        distribution_cfg=ActorHeadCfg.GaussianDistributionCfg(init_std=1.0),
    )
    networks.critic = RecurrentCriticHeadCfg(
        input=["critic", "privileged", "command"],
        hidden_dims=[256, 128],
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
```

You can replace either head independently. Keep `symmetry_cfg=None` for recurrent policies, as
required by the pinned upstream PPO implementation. Intermediate terms such as `state_estimator`
remain feed-forward. For your own architecture, see [the custom-term guide](docs/custom_terms.md):
subclass `TensorNetwork` and select it through `TensorNetworkCfg(class_name="your.module:YourModel", ...)`.

## Swapping loss types and functions in IsaacLab

Loss variants below use the same complete base config. `trains` selects the network parameters the
supervised loss may update; `prediction` selects a network output; `target` selects the label group.
Prediction and target shapes must match. Changing a loss function does not change the network or
create new labels.

### Swap MSE for Huber

```python
@configclass
class MyHuberPPORunnerCfg(MyCompositionalPPORunnerCfg):
    loss = MyCompositionalPPORunnerCfg().loss
    loss.estimator_loss.loss_fn = "huber"
    loss.estimator_loss.params = {"delta": 0.5}
    loss.estimator_loss.weight = 0.1
```

Built-in choices are `mse`, `huber`, `bce`, and `bce_with_logits`. For binary targets, use
`bce_with_logits` with an unconstrained network output; `bce` requires probability outputs in
`[0, 1]`. Neither is a replacement for continuous regression without also changing the target's
meaning. Clear or replace `params` when changing the criterion, since function arguments differ.

### Use a reusable function by import path

Cosine distance is useful when the direction of the supervised vectors is meaningful. It is
provided in the library's `components/loss` collection:

```python
@configclass
class MyCosinePPORunnerCfg(MyCompositionalPPORunnerCfg):
    loss = MyCompositionalPPORunnerCfg().loss
    loss.estimator_loss.loss_fn = "compositional_rsl_rl.components.loss:cosine_distance"
    loss.estimator_loss.params = {"eps": 1.0e-8}
```

### Supply your own function

A criterion takes `(prediction, target, **params)` and returns one finite scalar tensor connected
to the prediction. This function can live directly in the same task config file:

```python
import torch


def scaled_absolute_error(prediction, target, scale=1.0):
    return scale * torch.abs(prediction - target).mean()


@configclass
class MyCustomLossPPORunnerCfg(MyCompositionalPPORunnerCfg):
    loss = MyCompositionalPPORunnerCfg().loss
    loss.estimator_loss.loss_fn = scaled_absolute_error
    loss.estimator_loss.params = {"scale": 0.5}
```

Alternatively, place the function in `my_project/mdp/loss.py` and set
`loss_fn="my_project.mdp.loss:scaled_absolute_error"`. Both callables and import paths are supported.
Targets are detached by the loss manager. If a loss needs trainable components, declare those as
network terms and select them through `trains`.

### Give supervision a separate optimizer

```python
@configclass
class MySeparateEstimatorPPORunnerCfg(MyCompositionalPPORunnerCfg):
    loss = MyCompositionalPPORunnerCfg().loss
    loss.estimator_loss.optimizer = "separate"
    loss.estimator_loss.learning_rate = 1.0e-3
```

With `optimizer="joint"`, PPO and the weighted supervised objective can both update the estimator;
leave the loss's `learning_rate=None` because PPO owns its optimizer. With `optimizer="separate"`,
the estimator is trained by its supervised loss using a separate Adam optimizer. PPO consumes its
features but does not update its parameters.

To remove supervision, set `loss.estimator_loss = None` in a variant with its own copied `loss`
config. The estimator can then learn through PPO. Setting only `weight=0` retains optimizer ownership;
a separately owned estimator with zero loss weight will not learn from PPO.

## Network terms

```python
from dataclasses import dataclass
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg, MLPNetworkCfg, NetworkCfg


@dataclass(kw_only=True)
class MyNetworksCfg(NetworkCfg):
    state_estimator: MLPNetworkCfg | None = None


networks = MyNetworksCfg(
    actor=ActorHeadCfg(input=["policy", "state_estimator"], hidden_dims=[256, 128]),
    critic=CriticHeadCfg(input=["critic", "privileged"], hidden_dims=[256, 128]),
    state_estimator=MLPNetworkCfg(input=["policy"], output_dim=3, hidden_dims=[128, 64]),
)
```

Input dimensions come from environment observations and preceding network outputs. Terms may be declared
in any order: the manager sorts dependencies and rejects cycles or missing inputs. Actor and critic output
sizes come from the environment action dimension and the scalar value output. Other terms require
`output_dim`. Use `None` to disable an optional term; referencing a disabled term is an error.

Actor/critic are terminal heads. An input named `critic` or `actor` refers to a raw environment observation
when that group exists; it never introduces a head-to-head dependency. Other network names must not collide
with observation groups. Supervision targets prefer raw observation groups when names overlap; prediction
names always select networks.

All terms use the upstream model protocol. `CNNNetworkCfg` accepts the same `cnn_cfg` as RSL-RL's CNNModel,
including mixed vector/image inputs. Learned components are registered once in `NetworkManager.terms`.
Referencing one term from both actor and critic shares its parameters, without duplicate optimizer steps
or duplicate normalization updates. A loss-only decoder stays in training checkpoints but is pruned from
actor deployment exports.

## Losses and optimizer ownership

```python
from dataclasses import dataclass
from compositional_rsl_rl.loss import LossesCfg, SupervisedLossCfg


@dataclass(kw_only=True)
class MyLossesCfg(LossesCfg):
    estimator_loss: SupervisedLossCfg | None = None


loss = MyLossesCfg(
    estimator_loss=SupervisedLossCfg(
        trains=["state_estimator"],
        prediction="state_estimator",
        target="privileged_state",
        loss_fn="mse",
        optimizer="joint",
        weight=1.0,
    ),
)
```

Built-in functions: `mse`, `huber`, `bce` for probability predictions, and `bce_with_logits`. A callable or
qualified function path can supply another criterion with signature `(prediction, target, **params)` and
must return a finite scalar tensor. Prediction/target shapes must match exactly; accidental broadcasting
is rejected. Targets are detached, including network-generated targets.

| Setting | Meaning |
| --- | --- |
| `trains=[...]` | Only those unfrozen prediction dependencies receive gradients from this loss. |
| `optimizer="joint"` | Add the weighted loss to PPO's objective. PPO may also train these terms through policy/value paths. |
| `optimizer="separate", learning_rate=1e-3` | Use a separate Adam optimizer and detach these terms' outputs on PPO paths. PPO cannot update their parameters. |
| `weight=0` | Skip the loss computation while retaining its declared optimizer ownership. |
| loss term `None` | Remove the loss and its ownership declaration. |

Unselected weights are evaluated as detached weights, preserving gradients into selected upstream terms.
Separate optimizers must have disjoint ownership. A joint loss cannot select a separately owned term.
Actor/critic heads remain owned by PPO. All backwards complete before any optimizer step. Shared PPO
parameters are clipped once as a combined group; each separate optimizer's parameters are clipped separately
with the same `max_grad_norm`. Joint losses use PPO's learning rate and schedule; separate Adam learning
rates remain at their configured values.

Losses operate on the original minibatch before symmetry modifies observations, and recurrent padding is
removed before comparing prediction and target. RND retains its upstream optimizer. Log entries are
`loss/<term_name>` and contain unweighted mean losses. Stateful loss modules and loss-owned parameters are
not supported; put trainable components in named network terms.

## Package structure

```text
src/compositional_rsl_rl/
├── __init__.py
├── network/
│   ├── __init__.py
│   ├── network.py       # NetworkManager and actor/critic graph views
│   ├── network_cfg.py   # Common network configuration contracts
│   ├── tensor_network.py # Adapter for custom structured-input encoders
│   └── export.py        # TorchScript and ONNX actor export
├── loss/
│   ├── __init__.py
│   ├── loss.py          # Loss evaluation and optimizer ownership
│   └── loss_cfg.py      # Named supervised loss configs
├── algorithm/
│   ├── __init__.py
│   ├── algorithm.py     # CompositionalPPO
│   └── algorithm_cfg.py # PPO settings
├── runner/
│   ├── __init__.py
│   └── runner_cfg.py    # Configuration for upstream OnPolicyRunner
└── components/                # Reusable network architectures and loss criteria
    ├── network/        # pointnet.py, transformer.py, attention_map.py, network_cfg.py
    └── loss/           # loss.py
```

Each domain exports its public classes from `__init__.py`; there is no simulator-specific module
in the installed library. The algorithm's import path remains `compositional_rsl_rl:CompositionalPPO`.
See [the simulator-independent example](examples/basic.py) for ordinary dataclass configuration.
Given an RSL-RL-compatible `env`, use:

```python
from dataclasses import asdict
from rsl_rl.runners import OnPolicyRunner
from compositional_rsl_rl.runner import CompositionalRunnerCfg

cfg = CompositionalRunnerCfg(networks=networks, loss=loss)
runner = OnPolicyRunner(env, asdict(cfg), device=env.device)
runner.learn(num_learning_iterations=cfg.max_iterations)
```

Use `@dataclass(kw_only=True)` with annotated fields and `field(default_factory=...)` for mutable
subclass defaults. In IsaacLab task files, its `@configclass` decorator can wrap subclasses of these
same configs; the full example preserves its configuration utilities without coupling the library to them.

## Reusable custom terms

Use `compositional_rsl_rl.components.network` for the bundled PointNet, Transformer, and AME terms, and
`compositional_rsl_rl.components.loss` for reusable criteria. Managers and common config contracts stay in
`compositional_rsl_rl.network` and `compositional_rsl_rl.loss`. Task-specific implementations can live in
`your_project.mdp` and be selected with `class_name="your_project.module:YourNetwork"` or a loss
function import path; no manager registration is needed.

See [the custom-term guide](docs/custom_terms.md) for input layouts, custom encoder/loss examples,
and export requirements, and [the complete perception config](examples/perception.py) for switching
PointNet/Transformer terms with standard dataclasses. The PointNet encoder retains the supplied
IsaacLab implementation's masked pooling behavior and attribution, with no simulator imports.

## Attention-based map encoding (AME)

`AttentionMapNetworkCfg` adds CNN map tokens and robot-state-conditioned cross-attention, with
optional global context. It accepts a structured `[B, H, W, 3]` XYZ map, or a flattened map with an
explicit `map_shape`. The other inputs are vector observations used to form the attention query.
Unlike the generic Transformer term, this term uses robot-state queries to attend to map tokens.

This complete IsaacLab config shares one AME term between actor and critic. The environment must
provide `policy`, `critic`, and `elevation_map` observation groups. No auxiliary labels are required.
Save it as a task agent config and select `MyCompositionalPPORunnerCfg` through the existing
`rsl_rl_cfg_entry_point`. The executable copy is [examples/isaaclab_attention_map.py](examples/isaaclab_attention_map.py).

```python
# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Complete IsaacLab AME agent config; policy, critic, and elevation_map are observation groups."""

from compositional_rsl_rl.algorithm import CompositionalPPOAlgorithmCfg
from compositional_rsl_rl.loss import LossesCfg
from compositional_rsl_rl.components.network import AttentionMapNetworkCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg, NetworkCfg
from compositional_rsl_rl.runner import CompositionalRunnerCfg
from isaaclab.utils.configclass import configclass


@configclass
class MyNetworksCfg(NetworkCfg):
    """Share one actor-visible AME term across the actor and critic."""

    map_encoder: AttentionMapNetworkCfg | None = None


@configclass
class MyCompositionalPPORunnerCfg(CompositionalRunnerCfg):
    """Select this class with the task's existing rsl_rl_cfg_entry_point."""

    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 500
    networks = MyNetworksCfg(
        actor=ActorHeadCfg(
            input=["policy", "map_encoder"],
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=ActorHeadCfg.GaussianDistributionCfg(init_std=1.0),
        ),
        critic=CriticHeadCfg(
            input=["critic", "map_encoder"],
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
        ),
        map_encoder=AttentionMapNetworkCfg(
            input=["elevation_map", "policy"],
            map_input="elevation_map",
            map_shape=(33, 21, 3),
            output_dim=64,
            feature_dim=64,
            num_heads=16,
            cnn_channels=16,
            cnn_downsample=True,
            cnn_normalization="batch",
            attach_global=False,
            hidden_dims=[],
            obs_normalization=False,
            frozen=False,
            pretrained_cfg=None,
        ),
    )
    loss = LossesCfg()  # PPO trains the encoder; no auxiliary labels are required.
    algorithm = CompositionalPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
    )
    logger = "wandb"
    wandb_project = "attention_map_ppo"
    experiment_name = "attention_map_ppo"
```

Declare `attach_global=True, output_dim=128` for global context with `feature_dim=64` and no readout.
Declare nonempty `hidden_dims` if you want a learned projection to a different `output_dim`.
`obs_normalization` applies only to query vectors. The XYZ map must contain finite values with a
consistent, task-defined coordinate frame and grid convention.

The shared query uses actor-visible observations; the critic adds its own inputs in its head.
The reference uses distinct actor/critic queries with partially shared weights, which this shared
feature configuration does not reproduce exactly. Separate AME terms can use distinct queries,
but their internal weights are independent. See [the AME guide](docs/attention_map.md) for sharing
semantics, diagnostic attention weights, checkpoints, exports, and reproduction limits, or use
[the simulator-independent example](examples/attention_map.py) outside IsaacLab.

## Recurrent heads

Swap only the term config:

```python
from compositional_rsl_rl.network import RecurrentActorHeadCfg, RecurrentCriticHeadCfg

networks.actor = RecurrentActorHeadCfg(
    input=["policy", "state_estimator"],
    hidden_dims=[128, 128, 128],
    rnn_type="lstm",  # or "gru"
    rnn_hidden_dim=256,
    rnn_num_layers=1,
)
networks.critic = RecurrentCriticHeadCfg(
    input=["critic", "privileged"],
    hidden_dims=[128, 128, 128],
    rnn_type="lstm",
    rnn_hidden_dim=256,
    rnn_num_layers=1,
)
```

Either or both heads may be recurrent. The extension retains upstream rollout hidden states, episode resets,
trajectory padding, and truncated training behavior. Intermediate terms are currently feed-forward.
Recurrent intermediate/shared terms are rejected because they need additional hidden-state storage beyond
the upstream actor/critic protocol. Upstream RSL-RL also rejects symmetry with recurrent heads. Custom
sequence objectives beyond masked pointwise supervised losses are outside this release.

## Pretraining, freezing, checkpoints, and export

Declare pretraining, freezing, and the disabled auxiliary loss directly in the config constructor.
This complete agent config derives from `CompositionalRunnerCfg`; it does not modify an existing
experiment config. It is also available as [examples/isaaclab_g1_frozen.py](examples/isaaclab_g1_frozen.py).

The estimator architecture and input width must match the checkpoint. Replace the checkpoint path
before training. Because `estimator_loss=None`, this config does not require `privileged_state`
labels; it still requires the `policy`, `command`, `critic`, and `privileged` observation groups.

```python
# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Full IsaacLab task config with a pretrained, frozen state estimator and no auxiliary labels."""

from compositional_rsl_rl.algorithm import CompositionalPPOAlgorithmCfg
from compositional_rsl_rl.loss import LossesCfg, SupervisedLossCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg, MLPNetworkCfg, NetworkCfg
from compositional_rsl_rl.runner import CompositionalRunnerCfg
from isaaclab.utils.configclass import configclass


@configclass
class MyNetworksCfg(NetworkCfg):
    """Actor and critic are peers of the learned state estimator."""

    state_estimator: MLPNetworkCfg | None = None


@configclass
class MyLossesCfg(LossesCfg):
    """Named loss terms are independent of network construction."""

    estimator_loss: SupervisedLossCfg | None = None


@configclass
class MyFrozenEstimatorPPORunnerCfg(CompositionalRunnerCfg):
    """Select through the task's existing rsl_rl_cfg_entry_point; keep train/play scripts unchanged."""

    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 500

    # Raw observation dependencies are inferred from each network term's input list.
    networks = MyNetworksCfg(
        actor=ActorHeadCfg(
            input=["policy", "state_estimator", "command"],
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=ActorHeadCfg.GaussianDistributionCfg(init_std=1.0),
        ),
        critic=CriticHeadCfg(
            input=["critic", "privileged", "command"],
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
        ),
        state_estimator=MLPNetworkCfg(
            input=["policy"],
            output_dim=3,
            hidden_dims=[256, 128],
            activation="elu",
            obs_normalization=False,
            frozen=True,
            pretrained_cfg={
                "path": "/path/to/checkpoint.pt",
                "state_dict_key": "network_state_dict",
                "prefix": "terms.state_estimator.",
            },
        ),
    )

    loss = MyLossesCfg(estimator_loss=None)
    algorithm = CompositionalPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Enable your RslRlSymmetryCfg only when its observation transform handles all supplied groups.
        # Upstream RSL-RL disallows symmetry for recurrent actor/critic models.
        symmetry_cfg=None,
    )

    logger = "wandb"
    wandb_project = "g1_29dof_rigid_rough_ppo"
    experiment_name = "g1_29dof_rigid_rough_frozen_estimator_ppo"
```

Select `MyFrozenEstimatorPPORunnerCfg` through your task's `rsl_rl_cfg_entry_point`. The actor and
critic train with PPO while the loaded estimator remains frozen. To train the pretrained estimator
instead, declare `frozen=False`; add a `SupervisedLossCfg(...)` in the `loss` constructor only if you
also want supervised training.

### How `pretrained_cfg` selects weights

This config initializes one network term from a saved checkpoint. In the example, it loads the
saved state estimator into the `state_estimator=MLPNetworkCfg(...)` term being constructed.

| Field | Meaning in this example |
| --- | --- |
| `path` | Path to the checkpoint file on disk. Replace `/path/to/checkpoint.pt` with your file. Relative paths are resolved from the training process's working directory. |
| `state_dict_key` | Selects the checkpoint's top-level `network_state_dict` entry, which contains the compositional networks' parameters and buffers. |
| `prefix` | Keeps only entries whose names start with `terms.state_estimator.`, then removes that prefix before loading them into this network term. |

A full compositional training checkpoint contains entries shaped like this:

```text
checkpoint.pt
└── network_state_dict
    ├── terms.actor.mlp.0.weight
    ├── terms.critic.mlp.0.weight
    ├── terms.state_estimator.mlp.0.weight
    ├── terms.state_estimator.mlp.0.bias
    └── ...
```

The loader opens `path`, selects `checkpoint["network_state_dict"]`, and extracts the estimator's
entries. For example:

```text
Saved key:       terms.state_estimator.mlp.0.weight
Prefix removed: terms.state_estimator.
Loaded key:                            mlp.0.weight
```

`terms` is the network manager's module container; `state_estimator` is the network's name in the
**saved** config. Include the trailing dot in the prefix. If the source network was named `encoder`,
use `"prefix": "terms.encoder."` even if the destination term is named `state_estimator`.
The destination network's architecture and input width must match the saved parameter shapes.

Both `state_dict_key` and `prefix` depend on how the file was saved:

| Saved file contents | Fields to declare inside `pretrained_cfg` |
| --- | --- |
| Full compositional runner checkpoint | `path`, `state_dict_key="network_state_dict"`, and the source term's `prefix`. |
| A term's state dict directly, saved with `torch.save(estimator.state_dict(), path)` | Only `path`; keys already have names such as `mlp.0.weight`. |
| A term's state dict wrapped as `{"model": estimator.state_dict()}` | `path` and `state_dict_key="model"`; omit `prefix`. |

`pretrained_cfg` initializes parameters and buffers. `frozen=True` separately prevents this term
from training. Full training resume, including optimizer state, uses the runner's checkpoint loading
mechanism. Loading uses `weights_only=True`, CPU mapping, and strict state-dict matching by default.

### Checkpoints and export

Normalization
buffers are part of the model state. Frozen terms stay in evaluation mode, do not update normalization
statistics, and remain differentiable with respect to trainable inputs. Resume must retain the same frozen
and optimizer-ownership configuration; incompatible resumes fail before loading. For deliberate transfer,
load network weights without optimizer state or use per-term pretrained configuration.

Checkpoints retain the usual actor/critic/optimizer entries and additionally store all network terms and
separate optimizer states. Changing architecture requires explicit transfer; the earlier prototype's
`nodes.*` checkpoint format is not silently remapped.

Ordinary RSL-RL 5.5.1 checkpoints with `actor_state_dict`, `critic_state_dict`, and
`optimizer_state_dict` also load directly into a compositional configuration containing only two
independent, unfrozen `MLPModel` heads named `actor` and `critic`, with no auxiliary losses.
Use the same architecture, activation, normalization, action order, and observation input order as
training; checkpoint tensors cannot identify a same-sized observation permutation. No conversion
command is needed:

```python
runner.load("model_10000.pt")  # Actor, critic, optimizer, iteration, and curriculum counter.
# Or transfer only the policy, leaving the optimizer and curriculum progress unchanged:
runner.load("model_10000.pt", load_cfg={"actor": True, "iteration": False})
```

The loader checks head keys/shapes and maps the legacy optimizer's actor-then-critic parameter
order into the graph's order. Full resume requires the same optimizer type with one parameter group.
Shared networks, extra terms, recurrent heads, and the old combined `model_state_dict` format are not
converted. Newly saved checkpoints use the native compositional format.

For compositional policies, environments exposing `common_step_counter` (directly or through
`env.unwrapped`) also save an `environment_training_state` checkpoint entry containing that exact
counter and `num_steps_per_env`. The counter counts vectorized control steps, without multiplying by
the number of environments or GPUs. No simulator import or runner subclass is required.

Loading with `load_cfg=None` or `load_cfg["iteration"]=True` restores this counter as well as the runner
iteration. For older checkpoints without the entry, it estimates `(iter + 1) * num_steps_per_env` using
the current rollout length and emits a warning. Iteration zero is saved after the first rollout in
RSL-RL 5.5.1. This estimate assumes unchanged rollout length and no earlier curriculum restarts;
checkpoints with the new entry restore the exact counter even if rollout length changes.

For weights-only transfer, set `load_cfg["iteration"]=False` alongside the desired model load flags;
this leaves the environment's counter unchanged. Environments without this counter and ordinary PPO
configs are unaffected. Only the counter is restored: time-based curricula apply it at their next
normal update. Terrain levels, episode state, random-number generators, and other curriculum state
are not saved or advanced by this integration.

### Optional terrain curriculum state

Enable `env_interface` to save TerrainImporter-style terrain difficulty alongside the training
counter. Terrain restoration is disabled by default, so playback and finetuning keep the destination
terrain setup. The implementation lives entirely in `compositional_rsl_rl/env_interface.py`;
Isaac Lab environment and manager classes need no changes.

```python
from compositional_rsl_rl.env_interface import EnvInterfaceCfg

# Save terrain state; keep the destination terrain when loading a checkpoint:
env_interface = EnvInterfaceCfg()  # restore_terrain=False
# Opt into exact terrain restoration when resuming the same training layout:
env_interface = EnvInterfaceCfg(restore_terrain=True)
# Equivalent explicit opt-in for a dictionary runner config:
runner_cfg["env_interface"] = {"restore_terrain": True}
```

Checkpoints store each rank's terrain levels, fixed terrain column assignments, terrain origin
grid/seed, optional rigid/MPM membership, and control/physics counters. With restoration explicitly
enabled, loading validates the layout before loading policy weights. It restores tensors in place and calls the environment's
normal reset to place robots and particles on the saved terrain, then refresh observations.
During that reset only the terrain instance's `update_env_origins` is temporarily suppressed;
other curricula run at the restored counter. The method is restored in `finally`, including on
reset failure. This supports curricula that advance terrain through `update_env_origins`; it does
not cover custom curricula that write terrain levels directly. Robot/particle motion, solver
history, RNG state, and arbitrary stateful curriculum terms are not checkpointed.

Distributed snapshots are collected by every rank after each PPO update. Rank-zero-only `save()`
performs no collective, so it cannot wait on workers that are not saving. This adds one object
collective per PPO update when enabled. Distributed manual saves must use a collected snapshot;
call `alg.env_interface.capture()` on **every rank** if you modify terrain state outside the
training loop. After loading an old checkpoint without terrain state, capture again to refresh the saved counters.
After disabling restoration, run an update or call `capture()` on every rank before a manual
distributed save.

Exact terrain restoration requires the same world size, environments per rank, terrain layout,
seed, column assignments, and rigid/MPM membership. Keep the remaining terrain configuration
unchanged as well: origin-grid validation cannot detect every geometry/material change. Keep the
default `restore_terrain=False` for playback or finetuning on a different layout. Policy, optimizer,
and training-counter loading retain their existing behavior; only terrain restoration/reset are
skipped. `load_cfg` with `iteration=False` skips environment restoration entirely.

If `environment_training_state.terrain_curriculum` is absent, terrain loading is a no-op: it does
not reset the environment, change terrain assignments/origins, or write the physics-step counter.
The algorithm still restores/estimates the control-step counter independently. It cannot recover
terrain levels that were never saved.

The runner uses the same `load()` call for training and playback. In Isaac Lab, append
`agent.env_interface.restore_terrain=true` only when resuming the same training layout. Playback
and terrain transfer need no override with the default config; no `isaaclab_rl` changes are required.

The unchanged runner exports one complete actor graph and its required raw inputs. The exported `obs` tensor
is the concatenation of flattened groups in `policy.obs_groups` order:

```python
packed = torch.cat([obs[name].flatten(start_dim=1) for name in policy.obs_groups], dim=-1)
```

Keep that order and the observation shapes in deployment metadata. Feed-forward JIT accepts batched inputs.
Recurrent JIT retains upstream batch-one internal state and its `reset()` method. Recurrent ONNX exposes
`h_in`/`h_out`, plus `c_in`/`c_out` for LSTM. ONNX uses the upstream runner's fixed dummy batch of one.

## Compatibility boundaries and maintenance

- Supported dependency: **RSL-RL `5.5.1+flashsac.1`**, the commit-pinned FlashSAC fork of 5.5.1. This is not a promise of compatibility
  with every past/future RSL-RL or IsaacLab version.
- Configs use standard dataclasses. Only the IsaacLab task example uses its `configclass` decorator
  to support its task configuration and CLI utilities. The inspected train/play/export backend selects
  the ordinary `OnPolicyRunner`, which resolves the configured algorithm class dynamically.
- The example changes task config and may require a supervision observation group; no train/play script
  edits or global import interception are required.
- PPO has no public loss hook in the pinned dependency. `algorithm/algorithm.py` therefore adapts one versioned PPO
  update loop. Rollout collection, GAE, storage, runner, RND, and symmetry use upstream code. Upgrade this
  integration deliberately and rerun the package tests when changing the dependency pin.
- Parameters are deduplicated by ownership; the old `share_cnn_encoders` flag is rejected. Share named terms
  through input references instead.
- Custom models must conform to the upstream MLPModel protocol. `TensorNetwork` supplies this
  adapter and export wrappers for single-input encoders; their operators must still support export.
  CUDA training, arbitrary custom criteria/models, and robot-learning convergence require task validation.

## Tests

```bash
python -m pip install -e '.[test]'
python -m pytest tests
```

The suite checks real upstream runner training/resume/inference, exact loss gradient routing, joint/separate
optimizer ownership, shared normalization, GRU/LSTM and mixed recurrent heads, image terms, independent
ONNX Runtime outputs/states, pretrained/frozen terms, RND/symmetry/autocast, CPU distributed synchronization,
ordinary RSL-RL checkpoint compatibility, simulator-blocked dataclass training in a fresh process,
and real IsaacLab config serialization when IsaacLab is installed. Terrain checkpoint tests cover
restored placement, reset cleanup, layout validation, weights-only transfer, and per-rank terrain
state under a rank-zero-only save.
