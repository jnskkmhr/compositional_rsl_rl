# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional terrain curriculum checkpointing through the environment's existing public attributes.

No simulator imports are required. This supports TerrainImporter-style environments whose
terrain progression calls ``terrain.update_env_origins`` during the normal environment reset.
It saves training progress, not robot, particle, solver, or random-number-generator state.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch


@dataclass(kw_only=True)
class EnvInterfaceCfg:
    """Enable terrain snapshots; opt into restoration only when resuming the same environment layout."""

    restore_terrain: bool = False


class EnvInterface:
    """Capture rank-local terrain curricula and restore them before the next rollout."""

    def __init__(self, env: object, *, restore_terrain: bool = False) -> None:
        """Validate the opt-in terrain interface without importing the simulator."""
        self.env = getattr(env, "unwrapped", env)
        self.restore_terrain = restore_terrain
        self.terrain = self.env.scene.terrain
        for name in ("terrain_levels", "terrain_types", "terrain_origins", "env_origins", "update_env_origins"):
            if getattr(self.terrain, name, None) is None:
                raise ValueError(f"Terrain checkpointing requires terrain.{name}.")
        if not hasattr(self.env, "common_step_counter"):
            raise ValueError("Terrain checkpointing requires env.common_step_counter.")
        self._snapshot = None

    @staticmethod
    def _distributed() -> bool:
        """Whether all ranks must participate in snapshot collection."""
        return torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1

    def _local_state(self) -> dict:
        """Copy this rank's progress onto the CPU without retaining live simulator tensors."""
        terrain = self.terrain
        generator = getattr(terrain.cfg, "terrain_generator", None)
        state = {
            "common_step_counter": int(self.env.common_step_counter),
            "terrain_levels": terrain.terrain_levels.detach().cpu().clone(),
            "terrain_types": terrain.terrain_types.detach().cpu().clone(),
            "terrain_origins": terrain.terrain_origins.detach().cpu().clone(),
            "terrain_seed": getattr(generator, "seed", None),
        }
        if hasattr(self.env, "_sim_step_counter"):
            state["sim_step_counter"] = int(self.env._sim_step_counter)
        if hasattr(terrain, "mpm_env_mask"):
            state["mpm_env_mask"] = terrain.mpm_env_mask.detach().cpu().clone()
        return state

    def capture(self) -> None:
        """Collect once per PPO update on every rank, never inside rank-zero-only checkpoint writing."""
        local = self._local_state()
        if self._distributed():
            states = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(states, local)
        else:
            states = [local]
        self._snapshot = {"version": 1, "ranks": states}

    def state_dict(self) -> dict:
        """Return the latest collective snapshot, or current state in a single-process run."""
        if not self._distributed():
            self.capture()
        if self._snapshot is None:
            raise RuntimeError("Call capture on every rank before saving terrain curriculum state.")
        return copy.deepcopy(self._snapshot)

    def validate(self, state: dict | None) -> dict | None:
        """Validate a checkpoint without changing environment or policy state."""
        if not self.restore_terrain or state is None:
            return None
        world_size = torch.distributed.get_world_size() if self._distributed() else 1
        rank = torch.distributed.get_rank() if self._distributed() else 0
        if state.get("version") != 1 or len(state.get("ranks", [])) != world_size:
            raise ValueError("Terrain checkpoint version/world size differs; disable restore_terrain for transfer.")
        saved = state["ranks"][rank]
        current = self._local_state()
        for name in ("terrain_levels", "terrain_types"):
            value = saved[name]
            if value.shape != current[name].shape or value.dtype != current[name].dtype:
                raise ValueError(
                    f"Terrain checkpoint {name} shape/dtype differs; disable restore_terrain for transfer."
                )
        if saved["terrain_seed"] != current["terrain_seed"] or not torch.equal(
            saved["terrain_origins"].cpu(), current["terrain_origins"]
        ):
            raise ValueError("Terrain checkpoint layout/seed differs; use the original terrain configuration.")
        saved_mask, current_mask = saved.get("mpm_env_mask"), current.get("mpm_env_mask")
        if (saved_mask is None) != (current_mask is None) or (
            saved_mask is not None and not torch.equal(saved_mask.cpu(), current_mask)
        ):
            raise ValueError("Terrain checkpoint rigid/MPM membership differs; disable restore_terrain for transfer.")
        # Column assignments are fixed by TerrainImporter at construction, including rigid/MPM topology.
        if not torch.equal(saved["terrain_types"].cpu(), current["terrain_types"]):
            raise ValueError("Terrain checkpoint column assignments differ; use the original environment layout.")
        levels = saved["terrain_levels"]
        if torch.any(levels < 0) or torch.any(levels >= self.terrain.terrain_origins.shape[0]):
            raise ValueError("Terrain checkpoint contains out-of-range levels.")
        return saved

    def load_state_dict(self, state: dict | None) -> None:
        """Restore curricula in place and reset fresh episodes without advancing terrain difficulty."""
        if state is None:
            return
        if not self.restore_terrain:
            self._snapshot = None
            return
        saved = self.validate(state)
        self._snapshot = None
        for name in ("terrain_levels", "terrain_types"):
            target = getattr(self.terrain, name)
            target.copy_(saved[name].to(target.device))
        self.env.common_step_counter = int(saved["common_step_counter"])
        if hasattr(self.env, "_sim_step_counter"):
            self.env._sim_step_counter = (
                saved["sim_step_counter"]
                if "sim_step_counter" in saved
                else self.env.common_step_counter * self.env.cfg.decimation
            )
        self.terrain.env_origins.copy_(
            self.terrain.terrain_origins[self.terrain.terrain_levels, self.terrain.terrain_types]
        )
        # Keep all other reset/curriculum behavior, including step-based schedules and MPM solver resets.
        # Restore the original attribute even if reset fails; never patch the class or curriculum manager.
        had_override = "update_env_origins" in vars(self.terrain)
        original = self.terrain.update_env_origins
        self.terrain.update_env_origins = lambda *args, **kwargs: None
        try:
            with torch.inference_mode():
                self.env.reset()
        finally:
            if had_override:
                self.terrain.update_env_origins = original
            else:
                del self.terrain.update_env_origins
        self._snapshot = copy.deepcopy(state)
