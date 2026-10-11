# Reusable network and loss terms

The installed library's term collection lives in `src/compositional_rl/mdp/`:

```text
mdp/
├── __init__.py
├── network/
│   ├── __init__.py
│   ├── pointnet.py
│   ├── transformer.py
│   ├── attention_map.py
│   └── network_cfg.py
└── loss/
    ├── __init__.py
    └── loss.py
```

The name follows the requested IsaacLab convention. These are learning terms with no simulator
imports, observation managers, or environment state. `network/` and `loss/` at the package root
contain the managers and common configuration contracts; `mdp/` contains reusable implementations
and their concrete configs. This keeps orchestration separate from model architecture and criteria.

Put reusable library terms here. Put task-specific code in your own installed package, for example
`my_project/mdp/network/patch.py` and `my_project/mdp/loss.py`. Configs resolve ordinary import paths;
you do not need to edit a registry, either manager, the runner, or this library.

## PointNet and Transformer

```python
from compositional_rl.mdp.network import PointNetNetworkCfg, TransformerNetworkCfg

pointnet = PointNetNetworkCfg(
    input=["points"],  # One observation tensor: [batch, num_points, 4].
    output_dim=32,  # Term output exposed to downstream networks.
    feature_dim=128,  # Pooled encoder feature width before the MLP readout.
    pointnet_hidden_dims=[64, 128],
    hidden_dims=[128],  # Readout hidden layers, followed by output_dim outputs.
)

transformer = TransformerNetworkCfg(
    input=["tokens"],  # One observation tensor: [batch, num_tokens, channels].
    output_dim=32,
    feature_dim=128,  # Must be divisible by num_heads.
    num_heads=4,
    num_layers=2,
    feedforward_dim=256,
    validity_channel=True,  # Final channel is validity > 0.5; exclude it from token data.
    use_positional_encoding=True,
    hidden_dims=[128],
)
```

PointNet expects XYZ in the first three channels and validity in the fourth. It applies the same
MLP to each point and takes a masked maximum. The encoder is adapted from the supplied IsaacLab
`contrib/velocity/config/g1_29dof_perception/agents/pointnet.py`, retaining its BSD-3-Clause attribution
and encoder parameter names. Its outputs are invariant to point order. Invalid XYZ values, including
NaN padding, are ignored; an all-invalid cloud has zero pooled features. The learned readout can
still produce nonzero outputs from those zero features.

The Transformer uses pre-norm multi-head self-attention, GELU feed-forward blocks, and masked mean
pooling. It has no dropout. With `validity_channel=False`, every token is valid and every channel is
data. Learned positional embeddings distinguish token indices; disable them for unordered point
sets. Invalid tokens cannot affect valid-token attention or pooling. All-invalid inputs produce
zero pooled features before the readout. Sequence length is fixed by the input observation shape.
Attention has quadratic cost in the number of tokens, so choose the token count deliberately.

These are feed-forward encoders of one observation, not recurrent policies. Token indices only
represent time if your environment explicitly supplies an ordered history. You can still use a
recurrent actor or critic downstream. Neither encoder normalizes raw input coordinates or validity
flags; `obs_normalization=True` is rejected. Vector observations can be normalized in the downstream
MLP actor/critic as usual. The RSL-RL readout requires at least one hidden layer.

Assign either config to a named field in your `NetworkCfg` subclass, then reference that field in
actor/critic `input` lists. See [the complete perception example](../examples/perception.py), which
switches encoders with `make_config("pointnet")` or `make_config("transformer")` and includes an
optional supervised loss. Its labels are three task-defined values in `perception_target`; the
library does not invent labels from the point cloud.

## Attention-based map encoding

For a map conditioned on robot-state vectors, use `AttentionMapNetworkCfg`. It implements CNN map
tokens and state-query cross-attention with multiple inputs. See [the AME guide](attention_map.md)
and [complete IsaacLab config](../examples/isaaclab_attention_map.py) for the full configuration,
input layouts, sharing semantics, and attention visualization. The single-input `TensorNetwork`
adapter remains available for other custom encoders.

## Implement a custom network

For a single structured tensor input, subclass `TensorNetwork` and implement `build_encoder`.
For example, save this in `my_project/mdp/network/patch.py`:

```python
import math
from torch import nn
from compositional_rl.network import TensorNetwork


class PatchEncoder(TensorNetwork):
    def build_encoder(self, input_shape, feature_dim, activation):
        return nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(math.prod(input_shape), feature_dim),
            nn.SiLU(),
        )
```

Configure it from any task package:

```python
from compositional_rl.network import TensorNetworkCfg

patch_encoder = TensorNetworkCfg(
    class_name="my_project.mdp.network.patch:PatchEncoder",
    input=["patch"],
    feature_dim=64,
    hidden_dims=[64],
    output_dim=32,
)
```

The encoder must map `[batch, *input_shape]` to `[batch, feature_dim]` using registered PyTorch
modules. The adapter handles flattened recurrent minibatches, the MLP readout, optional action
distributions, and the existing model interface. It also wraps the complete term for flattened-input
TorchScript/ONNX export. The encoder's own operators must support those exporters; arbitrary Python
code is not automatically exportable. Export input shapes remain fixed to the observation schema.

For extra constructor options, extend `TensorNetworkCfg` with a standard dataclass and accept those
options in your subclass before calling `super().__init__`. For more general multi-input or custom
state behavior, use the upstream `MLPModel` protocol directly and implement its export contract;
`TensorNetwork` intentionally handles one input. Recurrent intermediate terms remain unsupported.

Per-term pretraining, freezing, sharing, and loss optimizer ownership work the same as for MLP terms.
The source PointNet encoder's weights can be loaded directly into `term.encoder`. To load a complete
term using `pretrained_cfg`, supply weights matching the new encoder/readout layout; the old combined
PointNetMLPModel checkpoint is not automatically remapped.

## Implement a custom loss

Loss functions are ordinary stateless functions. For example, in `my_project/mdp/loss.py`:

```python
import torch


def absolute_error(prediction, target):
    return torch.abs(prediction - target).mean()
```

Then use the common loss config:

```python
from compositional_rl.loss import SupervisedLossCfg

loss = SupervisedLossCfg(
    trains=["perception"],
    prediction="perception",
    target="perception_target",
    loss_fn="my_project.mdp.loss:absolute_error",
    optimizer="joint",
    weight=0.1,
)
```

The bundled `compositional_rl.mdp.loss:cosine_distance` is another example, useful when the direction
of feature vectors is meaningful. It accepts `params={"eps": 1.0e-8}`. A direct callable is also
accepted in `loss_fn`. Criteria receive matching prediction/target shapes after recurrent padding
is removed; targets are detached. Return one finite scalar tensor connected to the prediction.
Trainable loss components belong in network terms, where checkpointing and ownership are explicit.
