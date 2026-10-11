# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared and separate terms must remain synchronized across independent workers."""

import datetime
import torch
from pathlib import Path

import pytest
from test_composition import Environment, config, rollout

from compositional_rl import CompositionalPPO


def _worker(rank: int, store: str, separate: bool) -> None:
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        store=torch.distributed.FileStore(store, 2),
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=45),
    )
    try:
        torch.manual_seed(100 + rank)
        env = Environment()
        cfg = config("gru", separate)
        cfg["multi_gpu"] = {"global_rank": rank, "world_size": 2}
        algo = CompositionalPPO.construct_algorithm(env.get_observations(), env, cfg, "cpu")
        algo.broadcast_parameters()
        rollout(algo, env)
        algo.update()
        for parameter in algo.networks.parameters():
            values = [torch.empty_like(parameter) for _ in range(2)]
            torch.distributed.all_gather(values, parameter)
            torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
        assert algo.networks.terms["estimator"].obs_normalizer.count == 32
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="PyTorch Gloo is unavailable")
@pytest.mark.parametrize("separate", [False, True])
def test_two_worker_training(tmp_path: Path, separate: bool) -> None:
    """Broadcast and reduce every term exactly once, including separate optimizer parameters."""
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path / "store"), separate), nprocs=2, join=True)
