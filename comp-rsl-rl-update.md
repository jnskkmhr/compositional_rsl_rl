# Compositional RSL-RL: FlashSAC ownership refactor

This update supersedes the earlier integration design. Core RSL-RL owns learning, replay and
collection. Compositional RSL-RL owns the reference network architecture. The subsequent requested
update integrates the main IsaacLab checkout and adds registered rigid G1 agent configs and symmetry.

## Implementation

- Core owns the native FlashSAC update, replay and off-policy runner. There are no
  `rsl_rl/models/flash_sac_model.py` or `rsl_rl/modules/flash_sac.py` files.
- Feature composition reuses `NetworkManager`. `ResidualNetworkCfg` is an ordinary named
  feature term, like the existing map encoder. `ActorHeadCfg` and `CriticHeadCfg` reference
  encoder outputs through `input` and configure distributions through `distribution_cfg`.
- Removed the unnecessary `network/head.py` and `components/network/distribution.py` wrappers,
  along with residual-specific actor/critic configs. No replacement wrapper file was added.
- `components/network/residual.py` owns residual features and normalized linear projections.
  `LinearNetwork` is only a readout, with no embedded backbone. It preserves normalized output
  weights and independent Q ensemble projections from the reference implementation.
- Squashed Gaussian and categorical return distributions extend the existing RSL-RL distribution
  interface in `modules/distribution.py`. They also work on the existing `MLPModel` directly.
- `MLPModel` has optional generic runtime-input width/data arguments; `MLP` supports an empty
  hidden-layer list for a linear head. No residual architecture is added to core.
- `critic_action_input="critic_encoder"` selects where the FlashSAC graph adapter concatenates
  actions, using `env.num_actions` for width. State observation groups remain unchanged.
  The default injection point is the critic head. Action-conditioned terms cannot be shared
  with the actor or detached by a separate optimizer.
- Actor, critic and each Q ensemble member have independent parameters. Auxiliary-loss ownership,
  symmetry, and pre-reset replay semantics remain on the existing update paths.

## Compatibility and scope

The development package versions remain `rsl-rl-lib==5.5.1+flashsac.2` and
`compositional_rsl_rl==0.2.0.dev1`. The compositional package now pins its RSL-RL dependency
to an immutable Git commit using standard dependency metadata. The FlashSAC branches are
published on GitHub; these are development builds, not upstream or PyPI releases. The newly introduced FlashSAC constructor/head paths
from the earlier implementation are superseded; no implicit architecture or compatibility fallback
is retained for them. Existing PPO model behavior is unchanged.

The main `/home/jkamohara/isaac/IsaacLab` checkout now supports off-policy train/play/export and
pre-reset history capture. Rigid G1 configs are registered under `rsl_rl_flash_sac_cfg_entry_point`.
The older isolated `IsaacLab-flash-sac` worktree is historical and was not updated in this step.
Existing unrelated changes in the main checkout were preserved.

The compositional package owns the exact RSL-RL Git revision requirement. IsaacLab retains an
unversioned direct RSL-RL dependency and pins only the compositional Git revision. Its lockfile
resolves both libraries from GitHub, so a fresh checkout installs with `uv sync --locked` without
local sibling repositories.

The existing core `Symmetry` extension now supports owned replay-pair augmentation and a common
policy mirror objective. Both native and compositional FlashSAC consume it through `symmetry_cfg`.
Current/successor observations and labels are transformed together, while rewards, boundary flags
and n-step discounts retain their pairing. Collection-time reward statistics are unchanged.
G1 defaults explicitly match Agile learner hyperparameters and use augmentation without mirror loss;
mirror loss remains an opt-in extension. Replay capacity differs for the larger observation history,
and the G1 example now matches Agile's action-support curriculum through the generic scheduler.

Single-device eager feed-forward learning is supported, including optional CUDA FP16 AMP.
Shared trainable actor/critic graph terms, recurrent replay, compile mode and intermediate empirical
observation normalizers remain explicitly unsupported. Shared frozen graph terms are supported.
Checkpoint restoration restarts replay and simulator trajectories; exact continuation and Agile
checkpoint conversion are not implemented. The IsaacLab G1 Agile task now supplies its native curricula and replay relabeling through the runner extension; One-S remains outside this integration. Terrain-state restoration is not implemented for this runner.

## Validation results

The graph-based correction passed **219 core tests** and **100 composition tests**, plus the added
checkpoint/action-routing regression in a 10-test focused FlashSAC run. Tests cover
standard MLP distribution selection, differentiable runtime action inputs, residual features with
ordinary PPO heads, replay/gradient ownership, checkpoint restoration and JIT/ONNX exports.
The pinned reference comparison maps graph parameters to the original layout and compares eight
updates, including model weights and normalization buffers. Native G1 checks exercise the registered
config with symmetry and forced timeouts. CPU and CUDA FP16 AMP reference comparisons passed eight
updates. Native G1 training, playback, JIT/ONNX and LEAPP export passed. Checkpoint metadata preserves
and validates the selected action-routing term.

Earlier integration validation also passed 39 observation tests, 3 native wrapper cases and 57
entrypoint/export cases. Two unrelated entrypoint failures were traced to pre-existing eager LEAPP
imports and missing optional `skrl`. IsaacLab's repository-wide changelog gate reports unrelated
pre-existing fragment problems; the touched fragment follows the repository format.

The graph/config/state layout supersedes the earlier development checkpoints. Automatic migration
of those checkpoints is not included. Existing PPO configurations and import aliases remain supported.

See [the core interface](../rsl_rl/docs/flash_sac.md) and
[composition ownership](docs/flash_sac.md) for the current configuration and limitations.

## Generic learning-parameter curriculum

Added `curriculum_cfg={parameter_name: SchedulerCfg(type=..., params=...)}` to the FlashSAC config.
Linear ramps and discrete milestones are evaluated by a parameter-independent utility in the existing
RSL-RL utils module. The runner supplies completed vectorized collection steps; FlashSAC applies
supported values to runtime state, optimizer groups or derived target entropy as appropriate.

Supported parameters are action support, learning rate, target tau, actor-update period (discrete
integers) and temperature-target sigma. Explicit learning-rate schedules replace cosine decay.
Unsupported parameters fail at construction. Schedule configuration and applied progress are
checkpointed; playback/export use the saved bound, while replay retains historical executed actions.
The G1 example enables the 5,000-step hold and subsequent 5,000-step ramp from one to four.
No new source files or model classes were added for curricula. PPO integration is outside this update.

Curriculum validation passed 230 core tests and 102 compositional tests. A native G1 smoke test
used a shortened ramp, reached action support four, and verified the applied step in its checkpoint.
Playback, JIT/ONNX and LEAPP export from that checkpoint passed. Source lint and edited-file
formatting passed; IsaacLab's global changelog check still reports unrelated existing fragment errors.
