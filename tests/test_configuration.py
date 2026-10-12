# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the public config API can train without importing any simulator."""

import subprocess
import sys
import textwrap
from pathlib import Path


def test_configured_training_without_simulator_imports() -> None:
    """Block simulator imports in a fresh process, then train a dataclass-configured runner."""
    package = Path(__file__).parents[1]
    script = textwrap.dedent(
        """
        import importlib.abc
        import runpy
        import sys
        from dataclasses import asdict

        class BlockSimulators(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'isaaclab', 'isaaclab_rl', 'isaacsim', 'mjlab', 'newton'}:
                    raise AssertionError(f'Simulator import attempted: {fullname}')

        sys.meta_path.insert(0, BlockSimulators())
        sys.path.insert(0, sys.argv[2])
        import torch
        from test_composition import Environment
        from rsl_rl.runners import OnPolicyRunner
        from compositional_rsl_rl.algorithm.algorithm_cfg import CompositionalPPOAlgorithmCfg
        from compositional_rsl_rl.loss.loss_cfg import LossesCfg, SupervisedLossCfg
        from compositional_rsl_rl.network.network_cfg import MLPNetworkCfg, NetworkCfg
        from compositional_rsl_rl.runner.runner_cfg import CompositionalRunnerCfg
        from compositional_rsl_rl.mdp.network import AttentionMapNetworkCfg, PointNetNetworkCfg, TransformerNetworkCfg
        from compositional_rsl_rl.mdp.loss import cosine_distance

        cfg = runpy.run_path(sys.argv[1])['make_config']()
        cfg.device = 'cpu'
        cfg.num_steps_per_env = 4
        cfg.algorithm.num_learning_epochs = 1
        cfg.algorithm.num_mini_batches = 2
        data = asdict(cfg)
        assert 'GaussianDistributionCfg' not in data['networks']['actor']
        runner = OnPolicyRunner(Environment(), data, log_dir=None, device='cpu')
        before = [p.detach().clone() for p in runner.alg.networks.terms['estimator'].parameters()]
        runner.learn(num_learning_iterations=1)
        after = list(runner.alg.networks.terms['estimator'].parameters())
        assert any(not torch.equal(a, b) for a, b in zip(before, after))
        assert runner.alg.actor(runner.env.get_observations()).shape == (4, 2)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(package / "examples/basic.py"), str(package / "tests")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
