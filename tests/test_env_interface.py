# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain difficulty survives checkpoints without advancing during restoration."""

import copy
import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_composition import Environment, config, rollout

from compositional_rsl_rl import CompositionalPPO
from compositional_rsl_rl.env_interface import EnvInterface
from rsl_rl.runners import OnPolicyRunner


class Terrain:
    def __init__(self):
        self.cfg = SimpleNamespace(terrain_generator=SimpleNamespace(seed=42))
        self.terrain_levels = torch.tensor([1, 2, 1, 2])
        self.terrain_types = torch.tensor([0, 0, 1, 1])
        self.terrain_origins = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
        self.env_origins = self.terrain_origins[self.terrain_levels, self.terrain_types].clone()
        self.mpm_env_mask = torch.tensor([True, True, False, False])

    def update_env_origins(self, *args, **kwargs):
        self.terrain_levels.add_(1).remainder_(4)
        self.env_origins.copy_(self.terrain_origins[self.terrain_levels, self.terrain_types])


class TerrainEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.scene = SimpleNamespace(terrain=Terrain())
        self.common_step_counter = 100
        self._sim_step_counter = 400
        self.cfg = SimpleNamespace(decimation=4)
        self.fail_reset = False
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        self.scene.terrain.update_env_origins()
        if self.fail_reset:
            raise RuntimeError("reset failure")
        self.root_positions = self.scene.terrain.env_origins.clone()
        self.schedule_counter = self.common_step_counter
        self.episode_length_buf.zero_()
        return self.get_observations(), {}

    def step(self, actions):
        self.common_step_counter += 1
        self._sim_step_counter += 4
        return super().step(actions)


def terrain_config():
    cfg = config()
    cfg["env_interface"] = {}
    return cfg


def test_runner_restores_terrain_and_fresh_episodes(tmp_path: Path):
    env = TerrainEnvironment()
    source = OnPolicyRunner(env, terrain_config(), log_dir=None, device="cpu")
    source.learn(1)
    source.save(str(tmp_path / "model.pt"))
    levels = env.scene.terrain.terrain_levels.clone()
    env.scene.terrain.terrain_levels.zero_()
    env.common_step_counter = env._sim_step_counter = 0
    target = OnPolicyRunner(env, terrain_config(), log_dir=None, device="cpu")
    levels_ref = env.scene.terrain.terrain_levels
    origins_ref = env.scene.terrain.env_origins
    target.load(str(tmp_path / "model.pt"))
    torch.testing.assert_close(levels_ref, levels)
    assert levels_ref is env.scene.terrain.terrain_levels
    assert origins_ref is env.scene.terrain.env_origins
    torch.testing.assert_close(
        env.root_positions, env.scene.terrain.terrain_origins[levels, env.scene.terrain.terrain_types]
    )
    assert env.common_step_counter == env.schedule_counter == 104
    assert env._sim_step_counter == 416
    assert env.reset_count == 1
    assert "update_env_origins" not in vars(env.scene.terrain)
    env.reset()
    torch.testing.assert_close(env.scene.terrain.terrain_levels, (levels + 1) % 4)
    target.learn(1)
    assert env.common_step_counter == 108


def test_transfer_and_missing_terrain_state():
    env = TerrainEnvironment()
    algo = CompositionalPPO.construct_algorithm(env.get_observations(), env, terrain_config(), "cpu")
    checkpoint = copy.deepcopy(algo.save())
    checkpoint["iter"] = 10
    env.common_step_counter = 0
    algo.load(checkpoint, {"actor": True, "iteration": False}, True)
    assert env.common_step_counter == 0 and env.reset_count == 0
    del checkpoint["environment_training_state"]["terrain_curriculum"]
    with pytest.warns(RuntimeWarning, match="no terrain curriculum state"):
        algo.load(checkpoint, None, True)
    assert env.common_step_counter == env.schedule_counter == 100
    torch.testing.assert_close(env.scene.terrain.terrain_levels, torch.tensor([1, 2, 1, 2]))


@pytest.mark.parametrize("field", ["terrain_levels", "terrain_origins", "terrain_types", "mpm_env_mask"])
def test_incompatible_layout_rejected_before_reset(field):
    env = TerrainEnvironment()
    interface = EnvInterface(env)
    state = interface.state_dict()
    tensor = state["ranks"][0][field]
    if field == "terrain_levels":
        tensor[0] = 100
    elif field == "mpm_env_mask":
        tensor.logical_not_()
    else:
        tensor.add_(1)
    with pytest.raises(ValueError, match="Terrain checkpoint"):
        interface.load_state_dict(state)
    assert env.reset_count == 0
    interface.restore_terrain = False
    interface.load_state_dict(state)  # Explicit transfer can ignore incompatible terrain state.
    assert env.reset_count == 0


def test_reset_failure_restores_terrain_method():
    env = TerrainEnvironment()
    interface = EnvInterface(env)
    state = interface.state_dict()
    env.fail_reset = True
    with pytest.raises(RuntimeError, match="reset failure"):
        interface.load_state_dict(state)
    assert "update_env_origins" not in vars(env.scene.terrain)
    env.scene.terrain.update_env_origins()
    torch.testing.assert_close(env.scene.terrain.terrain_levels, torch.tensor([2, 3, 2, 3]))


def _distributed_worker(rank, store, checkpoint_path):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        store=torch.distributed.FileStore(store, 2),
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=45),
    )
    try:
        env = TerrainEnvironment()
        cfg = terrain_config()
        cfg["multi_gpu"] = {"global_rank": rank, "world_size": 2}
        algo = CompositionalPPO.construct_algorithm(env.get_observations(), env, cfg, "cpu")
        rollout(algo, env)
        env.scene.terrain.terrain_levels.fill_(rank + 1)
        algo.update()  # Every rank contributes its state here.
        if rank == 0:
            state = algo.save()  # Must not perform a collective here.
            state["iter"] = 0
            torch.save(state, checkpoint_path)
        torch.distributed.barrier()
        env.scene.terrain.terrain_levels.zero_()
        saved = torch.load(checkpoint_path, weights_only=False)
        algo.load(saved, None, True)
        torch.testing.assert_close(env.scene.terrain.terrain_levels, torch.full((4,), rank + 1))
        assert env.common_step_counter == 104 and env.reset_count == 1
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="PyTorch Gloo is unavailable")
def test_distributed_rank_zero_save_preserves_each_rank(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_worker, args=(str(tmp_path / "store"), str(tmp_path / "model.pt")), nprocs=2, join=True
    )
