from pathlib import Path

import numpy as np
import pytest

from f3dasm import ExperimentSample
from f3dasm._src.experimentdata.experimentsample import JobStatus
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


@pytest.fixture
def sample_domain() -> Domain:
    domain = Domain()
    domain.add_float(name='x1')
    domain.add_float(name='x2')
    domain.add_output(name='y1')

    return domain


def test_initialization(sample_domain):
    sample = ExperimentSample(
        input_data={'x1': 1, 'x2': 2},
        output_data={'y1': 3},
        domain=sample_domain,
        job_status='FINISHED',
        project_dir=Path("/tmp")
    )

    assert sample.input_data == {'x1': 1, 'x2': 2}
    assert sample.output_data == {'y1': 3}
    assert sample.job_status == JobStatus.FINISHED
    assert sample.project_dir == Path("/tmp")


def test_default_initialization(sample_domain):
    sample = ExperimentSample(domain=sample_domain)
    assert sample.input_data == {}
    assert sample.output_data == {}
    assert sample.job_status == JobStatus.OPEN
    assert sample.project_dir == Path.cwd()


def test_from_numpy(sample_domain):
    array = np.array([1.0, 2.0])
    sample = ExperimentSample.from_numpy(array, domain=sample_domain)
    assert sample.input_data == {'x1': 1.0, 'x2': 2.0}
    assert sample.output_data == {}


def test_from_numpy_no_domain():
    array = np.array([1.0, 2.0])
    sample = ExperimentSample.from_numpy(array)
    assert sample.input_data == {'x0': 1.0, 'x1': 2.0}
    assert sample.output_data == {}
    expected_domain = Domain()
    expected_domain.add_float(name='x0')
    expected_domain.add_float(name='x1')
    assert sample.domain == expected_domain


def test_addition(sample_domain):
    sample1 = ExperimentSample(input_data={'x1': 1}, output_data={
                               'y1': 3}, domain=sample_domain)
    sample2 = ExperimentSample(
        input_data={'x2': 2}, output_data={}, domain=sample_domain)
    combined = sample1 + sample2
    assert combined.input_data == {'x1': 1, 'x2': 2}
    assert combined.output_data == {'y1': 3}


def test_equality(sample_domain):
    sample1 = ExperimentSample(input_data={'x1': 1}, output_data={
                               'y1': 3}, domain=sample_domain)
    sample2 = ExperimentSample(input_data={'x1': 1}, output_data={
                               'y1': 3}, domain=sample_domain)
    assert sample1 == sample2


def test_replace_nan(sample_domain):
    sample = ExperimentSample(input_data={'x1': float('nan')}, output_data={
                              'y1': 3}, domain=sample_domain)
    sample.replace_nan(0)
    assert sample.input_data == {'x1': 0}


def test_round(sample_domain):
    sample = ExperimentSample(input_data={'x1': 1.2345}, output_data={
                              'y1': 3.6789}, domain=sample_domain)
    sample.round(2)
    assert sample.input_data == {'x1': 1.23}
    assert sample.output_data == {'y1': 3.68}


def test_store(sample_domain):
    sample = ExperimentSample(domain=sample_domain)
    sample.store('y2', 42, to_disk=False)
    assert sample.output_data['y2'] == 42


def test_mark(sample_domain):
    sample = ExperimentSample(domain=sample_domain)
    sample.mark('in_progress')
    assert sample.job_status == JobStatus.IN_PROGRESS


def test_to_multiindex(sample_domain):
    sample = ExperimentSample(input_data={'x1': 1}, output_data={
                              'y1': 3}, domain=sample_domain)
    multiindex = sample.to_multiindex()
    assert multiindex == {
        ('jobs', ''): 'finished',
        ('input', 'x1'): 1,
        ('output', 'y1'): 3
    }


def test_to_numpy(sample_domain):
    sample = ExperimentSample(input_data={'x1': 1, 'x2': 2}, output_data={
                              'y1': 3}, domain=sample_domain)
    input_array, output_array = sample.to_numpy()
    np.testing.assert_array_equal(input_array, [1, 2])
    np.testing.assert_array_equal(output_array, [3])


def test_to_dict(sample_domain):
    sample = ExperimentSample(input_data={'x1': 1, 'x2': 2}, output_data={
                              'y1': 3}, domain=sample_domain)
    dictionary = sample.to_dict()
    assert dictionary == {'x1': 1, 'x2': 2, 'y1': 3}


def test_invalid_status(sample_domain):
    sample = ExperimentSample(domain=sample_domain)
    with pytest.raises(ValueError):
        sample.mark('invalid_status')


def test_get(sample_domain):
    sample = ExperimentSample(input_data={'x1': 1}, output_data={
                              'y1': 3}, domain=sample_domain)
    assert sample.get('x1') == 1
    assert sample.get('y1') == 3
    with pytest.raises(KeyError):
        sample.get('nonexistent')


def test_is_status(sample_domain):
    """
    Test the is_status method of the ExperimentSample class.

    Parameters
    ----------
    sample_domain : Domain
        The domain fixture for the experiment sample.
    """
    sample = ExperimentSample(domain=sample_domain, job_status='OPEN')
    assert sample.is_status('open')
    assert not sample.is_status('finished')


def test_store_to_disk(sample_domain, tmp_path):
    """
    Test the store method of the ExperimentSample class with to_disk=True.

    Parameters
    ----------
    sample_domain : Domain
        The domain fixture for the experiment sample.
    tmp_path : Path
        Temporary directory for storing the object.
    """
    def dummy_store_function(obj, path):
        with open(path, 'w') as f:
            f.write(str(obj))
        return path

    def dummy_load_function(path):
        with open(path, 'r') as f:
            return f.read()

    sample = ExperimentSample(domain=sample_domain, project_dir=tmp_path)
    sample.store('y2', 42, to_disk=True, store_function=dummy_store_function,
                 load_function=dummy_load_function)
    assert sample._output_data['y2'] == 42
    assert sample.domain.output_space['y2'].to_disk
    assert sample.domain.output_space['y2'].store_function == dummy_store_function
    assert sample.domain.output_space['y2'].load_function == dummy_load_function


def test_to_numpy_with_nan(sample_domain):
    """
    Test the to_numpy method of the ExperimentSample class with NaN values.

    Parameters
    ----------
    sample_domain : Domain
        The domain fixture for the experiment sample.
    """
    sample = ExperimentSample(input_data={'x1': float('nan'), 'x2': 2}, output_data={
                              'y1': 3}, domain=sample_domain)
    input_array, output_array = sample.to_numpy()
    np.testing.assert_array_equal(input_array, [np.nan, 2])
    np.testing.assert_array_equal(output_array, [3])
