import pytest
from omegaconf import OmegaConf

from f3dasm._src.experimentsample import ExperimentSample
from f3dasm._src.hydra_utils import update_config_with_experiment_sample
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


@pytest.fixture
def sample_config():
    return OmegaConf.create(
        {"parameter1": 1, "parameter2": 2, "nested": {"parameter3": 3}}
    )


@pytest.fixture
def experiment_sample():
    return ExperimentSample(
        _input_data={"parameter1": 10, "nested.parameter3": 30},
        _output_data={"parameter2": 20},
    )


def test_update_config_with_experiment_sample(
    sample_config, experiment_sample
):
    updated_config = update_config_with_experiment_sample(
        sample_config, experiment_sample
    )

    assert updated_config.parameter1 == 10
    assert updated_config.parameter2 == 20
    assert updated_config.nested.parameter3 == 30


def test_update_config_skips_invalid_keys():
    """Keys that cause AttributeError should be silently skipped."""
    config = OmegaConf.create({"a": 1})
    # Key "nonexistent.deep.path" will cause AttributeError
    # when force_add=False (default)
    sample = ExperimentSample(
        _input_data={"nonexistent.deep.path": 99},
        _output_data={},
    )
    result = update_config_with_experiment_sample(config, sample)
    # Should not raise; original 'a' unchanged
    assert result.a == 1
