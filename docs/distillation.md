# Multiple teachers with PPO

`CompositionalPPO` supports supervised imitation of several frozen policies in the
same update as PPO. Each teacher is a named network; each imitation term is a
`DistillationLossCfg` under the existing runner `loss` configuration. There is no
second student optimizer or separate copy of PPO's update loop.

See [the complete configuration](../examples/multi_teacher_distillation.py).
Serialize it with `dataclasses.asdict` and pass it to the ordinary RSL-RL
`OnPolicyRunner`, with an environment implementing the usual `VecEnv` interface.

## What is optimized

For each minibatch, the student and teachers see the **same stored environment
states**, through their respective observation groups. The student supplies its
deterministic action mean, and each teacher supplies its deterministic action
mean. The loss added before PPO's backward pass is:

```
PPO loss + sum(weight_i * mean((student_actions[indices_i] - teacher_i_actions)**2))
```

`prediction_indices` selects student action components in the specified order;
`target_indices` optionally selects/reorders teacher action components. `None`
uses all components. Indices are unique, nonnegative integers on the last tensor
axis. Selected output shapes must match. Each teacher's MSE is averaged over its
selected components and minibatch, so weights are per teacher, not per joint.
Overlapping mappings intentionally add competing losses; unmapped joints receive
only PPO and any other configured losses. Set a term's `weight=0.0` to disable its
contribution. `loss_fn="huber"` is also supported.

This is on-policy PPO with auxiliary distillation: **the student controls the
rollout**, and the environment reward still trains PPO. It is not a teacher-action
rollout, a teacher-selection planner, or a stand-alone behavior-cloning runner.

## Teacher inputs and checkpoints

`TeacherPolicyCfg` defaults to a frozen MLP with a Gaussian distribution. Match
its architecture, observation normalization, distribution type, output width, and
input ordering to the source policy. `pretrained_cfg` uses the existing loader:

```python
pretrained_cfg={"path": "/path/to/model.pt", "state_dict_key": "actor_state_dict"}
```

For a compositional checkpoint term:

```python
pretrained_cfg={
    "path": "/path/to/composed.pt",
    "state_dict_key": "network_state_dict",
    "prefix": "terms.actor.",
}
```

The latter loads one term, not an arbitrary multi-term actor graph. For such a
teacher, declare and freeze its encoder/dependency terms too and load each term
by prefix. Teacher policies must be feed-forward; recurrent intermediate networks
remain unsupported. Recurrent **student** actor/critic heads use the existing
rollout hidden-state and padding handling.

Teacher distributions are retained so actor checkpoints load strictly, including
their standard deviations; teacher inference does not sample them. All teacher
parameters and normalizers remain fixed, including during PPO's normalization
update. Every dependency of a distillation target must be frozen; an unfrozen
teacher encoder is rejected.

The environment must supply teacher observation groups during training. Reproduce
the teachers' original observation order, scaling, history, joint order, coordinate
frames, and command meanings. The library does not manufacture locomotion velocity
commands or wrist SE(3) targets. For the planned box-pushing task, those are future
environment terms: robot velocity targets and wrist targets on the box face.
Do not feed the box velocity directly to an incompatible teacher observation layout.

Action mappings select columns; they do **not** convert action scales, offsets,
units, or joint conventions. These must agree between student and teacher, or be
converted explicitly with a custom criterion/network. Shared student encoders must
be listed in `trains` as well as `actor` if they should receive distillation gradients.

## Checkpoints, export, and distributed training

Full training checkpoints save the student, all teacher weights and normalization
buffers, optimizer state, and teacher/action mappings. A full optimizer resume
rejects changed mappings. For a deliberate new mapping, load network weights
without optimizer state using the existing partial-load API.

When resuming from a full checkpoint, set the teachers' `pretrained_cfg=None`
before constructing the runner, then load the full checkpoint. This avoids a
dependency on the original teacher files. Do not begin fresh training with those
randomly initialized teacher terms.

Teacher networks are not dependencies of the student actor in the example.
Therefore student export/inference needs only the student's `policy` group and
contains no teacher policies. Teacher networks are included in the existing
initial parameter broadcast; student distillation gradients are reduced together
with PPO gradients before the optimizer step.

Unconfigured distillation leaves the existing PPO path unchanged. Old checkpoints
without mapped/distillation losses remain loadable. No task, simulator, CUDA, or
Newton changes are required for this library feature.

## Validation

Run from this repository with its pinned RSL-RL 5.5.1 dependency:

```bash
uv run --extra test python -m pytest tests/test_distillation.py -q
uv run --extra test python -m pytest tests -q
```

The distillation tests exercise actual PPO rollouts/updates, independent actor
checkpoint loading, frozen normalization buffers, noncontiguous action mappings,
recurrent students, two-worker CPU/Gloo training, full-checkpoint resume, mapping
mismatch rejection, and student-only TorchScript export. Training logs report each
unweighted teacher criterion as `loss/<term_name>`; the configured weight is
applied only when adding it to the total optimization loss.

## Transition from imitation to PPO

PPO's `curriculum_cfg` accepts schedules for `policy_loss_weight` and
`loss.<term_name>.weight`. Schedules set absolute coefficients (not multipliers
on the original configured loss weight). Reuse the existing `SchedulerCfg` or
its dictionary representation:

```python
algorithm.curriculum_cfg = {
    "policy_loss_weight": {
        "type": "linear",
        "params": {"start_step": 48_000, "end_step": 240_000, "start_value": 0.1, "end_value": 1.0},
    },
    "loss.locomotion.weight": {
        "type": "linear",
        "params": {"start_step": 48_000, "end_step": 240_000, "start_value": 1.0, "end_value": 0.1},
    },
    "loss.manipulation.weight": {
        "type": "linear",
        "params": {"start_step": 48_000, "end_step": 240_000, "start_value": 1.0, "end_value": 0.1},
    },
}
```

The objective is `policy_loss_weight * (surrogate - entropy_coef * entropy)
+ value_loss_coef * value_loss + sum(weight_i * distillation_i)`, plus the
existing independently weighted symmetry/RND terms. The critic continues
learning while policy improvement is small. Loss magnitudes and gradients differ;
these coefficients do not guarantee a particular ratio of parameter updates.

Steps count completed rollout control steps **per environment**, advancing once
after each successful PPO update by `num_steps_per_env`. They do not count
minibatches, physics substeps, or total samples across environments/GPUs. Thus the
example holds its initial weights for 2,000 updates and reaches its final weights
at 10,000 updates when `num_steps_per_env=24`. Linear schedules clamp outside
their interval; `type="discrete"` with `params={"steps": [...], "values": [...]}`
provides abrupt stages instead.

Weights remain fixed throughout each PPO update and are reported as
`weight/policy` and `weight/<loss_name>` in the runner loss metrics. Zero-weight
teacher terms are not evaluated. Ordinary checkpoint resume restores schedule
progress; a weights-only load with `iteration=False` retains the new run's
progress. Older checkpoints without the counter infer it from `(iter + 1) *
num_steps_per_env`, or start at zero if no iteration metadata exists. Schedule
definitions come from the current configuration, so changing them on resume
intentionally changes the coefficients at the restored step.

With no curriculum and the default `policy_loss_weight=1.0`, existing fixed-weight
PPO behavior is preserved.
