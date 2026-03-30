"""Tests for SlurmResources and SlurmCluster dataclasses."""

import pytest
from omegaconf import OmegaConf

from f3dasm._src.pipeline.resources import SlurmCluster, SlurmResources

pytestmark = pytest.mark.smoke


class TestSlurmResources:
    def test_default_values(self):
        r = SlurmResources()
        assert r.time == "01:00:00"
        assert r.mem == "4G"
        assert r.cpus_per_task == 1
        assert r.nodes == 1
        assert r.max_array_size == 900
        assert r.max_concurrent == 64
        assert r.extra_sbatch == {}

    def test_custom_values(self):
        r = SlurmResources(
            time="02:00:00",
            mem="16G",
            cpus_per_task=4,
            nodes=2,
            max_array_size=500,
            max_concurrent=32,
            extra_sbatch={"gres": "gpu:1"},
        )
        assert r.time == "02:00:00"
        assert r.mem == "16G"
        assert r.cpus_per_task == 4
        assert r.nodes == 2
        assert r.extra_sbatch == {"gres": "gpu:1"}


class TestSlurmCluster:
    def test_default_values(self):
        c = SlurmCluster()
        assert c.partition == "batch"
        assert c.account == "default"
        assert c.env_setup == []
        assert c.env_vars == {}
        assert c.runner == "python"
        assert c.log_dir == "logs/{project_job}"

    def test_custom_values(self):
        c = SlurmCluster(
            partition="gpu",
            account="myaccount",
            env_setup=["module load cuda"],
            env_vars={"CUDA_VISIBLE_DEVICES": "0"},
            runner="uv run",
        )
        assert c.partition == "gpu"
        assert c.account == "myaccount"
        assert c.env_setup == ["module load cuda"]
        assert c.env_vars == {"CUDA_VISIBLE_DEVICES": "0"}
        assert c.runner == "uv run"

    def test_from_yaml(self):
        config = OmegaConf.create(
            {
                "partition": "compute",
                "account": "proj123",
                "env_setup": ["module load python"],
                "env_vars": {},
                "runner": "python",
                "log_dir": "logs/{project_job}",
            }
        )
        c = SlurmCluster.from_yaml(config)
        assert c.partition == "compute"
        assert c.account == "proj123"
        assert c.env_setup == ["module load python"]

    def test_from_yaml_strips_enabled_key(self):
        config = OmegaConf.create(
            {
                "enabled": True,
                "partition": "batch",
                "account": "default",
                "env_setup": [],
                "env_vars": {},
                "runner": "python",
                "log_dir": "logs/{project_job}",
            }
        )
        c = SlurmCluster.from_yaml(config)
        assert c.partition == "batch"
        # "enabled" should not cause an error
