# Attention-based map encoding (AME)

`AttentionMapNetwork` combines an XYZ elevation map with robot-state observation vectors:

```text
XYZ map → CNN → map tokens ───────────────→ attention keys and values
robot-state vectors → query projection ──→ attention query
                                               ↓
                                      map feature vector
                                               ↓
                                     actor / critic heads
```

The design follows the CNN/state-query cross-attention architecture in
[SII-FUSC/AME_Locomotion](https://github.com/SII-FUSC/AME_Locomotion), including its optional
CNN downsampling and global-context branch. This implementation uses standard PyTorch layers and
explicit compositional inputs; it does not import that repository's combined `ActorCriticEncoder`.
The reference's architecture and configuration are described in its
[encoder source](https://github.com/SII-FUSC/AME_Locomotion/blob/main/rsl_rl/rsl_rl/modules/actor_critic_encoder.py).

## Inputs and configuration

```python
from compositional_rl.mdp.network import AttentionMapNetworkCfg

map_encoder = AttentionMapNetworkCfg(
    input=["elevation_map", "policy"],
    map_input="elevation_map",
    map_shape=(33, 21, 3),
    feature_dim=64,
    num_heads=16,
    output_dim=64,
    cnn_channels=16,
    cnn_downsample=True,
    cnn_normalization="batch",
    attach_global=False,
    hidden_dims=[],
    obs_normalization=False,
)
```

The environment supplies:

| Input | Shape | Meaning |
| --- | --- | --- |
| `elevation_map` | `[B, H, W, 3]` | XYZ coordinates for each elevation-map grid cell. |
| `policy` | `[B, P]` | Robot-state observations used to form the attention query. |

The map may instead be flattened to `[B, H * W * 3]` when `map_shape=(H, W, 3)` is explicit.
For a structured map, `map_shape=None` infers its shape. Channel-first maps and height-only maps
are not accepted by this term. The task must supply finite XYZ values in a documented coordinate
frame and use the same origin, units, grid indexing, and preprocessing at training and deployment.
The library does not infer a coordinate frame or convert world coordinates into robot coordinates.
There is no validity-mask channel; handle missing map samples in the environment before encoding.

`map_input` identifies exactly one entry in `input`. Every other entry is a vector query input,
concatenated in its declared order. For example, `input=["policy", "elevation_map", "command"]`
uses `policy` and `command` together as the query. Query inputs may also reference outputs of other
feed-forward network terms. Enabling `obs_normalization` normalizes these query vectors only;
XYZ map coordinates remain unchanged.

`feature_dim` must be positive and divisible by `num_heads`. The CNN uses two convolution layers;
`cnn_downsample=True` gives the first layer stride two. Both odd and even map sizes are supported.
Its output grid is `[ceil(H / 2), ceil(W / 2)]` when downsampling and `[H, W]` otherwise. Convolution
normalization is `"batch"` by default, following the reference architecture; `"none"` disables it.
BatchNorm has its normal training-time batch dependence and running statistics. A frozen term keeps
these statistics fixed. Use `"none"` when batch-dependent features are inappropriate for your task.

Without global context the encoder returns `feature_dim` features. With `attach_global=True`, an
MLP processes map tokens and max-pools them to form global context. That context is included in the
query and concatenated with the attended features, producing `2 * feature_dim` outputs. For example:

```python
map_encoder = AttentionMapNetworkCfg(
    input=["elevation_map", "policy"],
    map_input="elevation_map",
    map_shape=(33, 21, 3),
    feature_dim=64,
    num_heads=16,
    attach_global=True,
    global_hidden_dims=[256, 128],
    output_dim=128,
    hidden_dims=[],
)
```

`hidden_dims=[]` uses the encoder output directly and validates its output dimension. To project
it to another width, set a nonempty readout, such as `hidden_dims=[128], output_dim=32`. `activation`
controls the global/readout MLPs; the CNN uses ReLU. The term has no action distribution or temporal
state. Attach an ordinary actor head, or a GRU/LSTM head, downstream.

## Sharing and training

See the [simulator-independent config](../examples/attention_map.py) and the
[complete IsaacLab config](../examples/isaaclab_attention_map.py). Both declare one `map_encoder`
term and reference it from actor and critic inputs. The term is registered once, its parameters
occur once in the optimizer, and its query normalizer is updated once per rollout update.
Sharing a term shares its parameters and buffers; it does not cache activations across separate
actor and critic forward calls. BatchNorm follows each forward call's normal behavior.

The shared query uses only actor-visible observations. The critic can receive additional privileged
observations through its own `input` list. The actor's exported graph does not acquire privileged
critic inputs through sharing. PPO policy/value gradients train the encoder; `loss=LossesCfg()`
requires no supervision target. Existing supervised losses, separate optimizer ownership, pretraining,
and freezing are also supported. Do not freeze a term that a configured loss still lists in `trains`.

The reference shares CNN and attention parameters while constructing different actor and critic
queries. The provided compositional config deliberately shares a single actor-visible query and
encoded feature vector. It does not reproduce the reference's distinct critic query. If you need a
privileged critic query, declare separate terms instead:

```python
actor_map = AttentionMapNetworkCfg(
    input=["elevation_map", "policy"],
    output_dim=64,
)
critic_map = AttentionMapNetworkCfg(
    input=["elevation_map", "critic"],
    output_dim=64,
)
```

Add these fields to your `NetworkCfg` subclass and reference `actor_map` only from the actor and
`critic_map` only from the critic. Their parameters are independent; naming two terms does not tie
their internal attention weights. Partial parameter tying between independently queried AME terms
is not implemented. Existing monolithic AME checkpoints require an explicit conversion; the
compositional parameter layout and head-input ordering differ.

## Attention visualization and export

Policy inference continues to return an action tensor. To inspect weights separately:

```python
runner.alg.eval_mode()
obs = env.get_observations().to(runner.device)
inputs = runner.alg.networks.features("map_encoder", obs)
weights = runner.alg.networks.terms["map_encoder"].attention_weights(inputs)
mean_weights = weights.mean(dim=1)
```

For a `[33, 21, 3]` map with stride-two downsampling, `weights` has shape `[B, num_heads, 17, 11]`;
`mean_weights` has shape `[B, 17, 11]`. Each head's weights sum to one over the CNN output grid.
The grid describes CNN map-token locations, rather than individual raw height samples. Its physical
orientation follows the environment's input-map indexing. Diagnostic calls require eval mode so
visualization does not update BatchNorm state, and they do not retain autograd graphs.

The complete actor graph supports the existing runner's TorchScript and ONNX exporters. Input groups
are flattened and concatenated in `policy.obs_groups` order as elsewhere in this library. The AME
term reconstructs its map and query inputs internally. Attention weights are available through the
Python diagnostic method; the default deployed actor retains its normal action-only output.

The tests compare cross-attention to PyTorch MHA, check map/query gradients and spatial weights,
train shared terms with joint/separate objectives and recurrent heads, exercise frozen checkpoint
transfer, and compare exported actors with ONNX Runtime. This establishes model integration, not
locomotion performance: observation construction, rewards, terrain curricula, and the reference's
two-stage training procedure remain environment/task responsibilities. CUDA robot training has not
been validated by these CPU tests.
