from typing import Iterable

import pandas as pd
import pytest

from f3dasm.design import NoOpenJobsError, Status, _JobQueue

pytestmark = pytest.mark.smoke


@pytest.fixture
def sample_job_queue():
    jobs = pd.Series(['open', 'open', 'in progress',
                     'finished'], dtype='string')
    job_queue = _JobQueue(jobs)

    yield job_queue

    # Reset the job queue to its original state after each test
    job_queue.reset()


@pytest.fixture
def empty_job_queue():
    return _JobQueue()


def test_job_queue_initialization(sample_job_queue: _JobQueue):
    assert isinstance(sample_job_queue.jobs, pd.Series)


def test_job_queue_repr_html(sample_job_queue: _JobQueue):
    assert isinstance(sample_job_queue._repr_html_(), str)


def test_job_queue_remove(sample_job_queue: _JobQueue):
    sample_job_queue.remove([1, 3])
    expected_jobs = pd.Series(['open', 'in progress'], index=[
                              0, 2], dtype='string')
    assert sample_job_queue.jobs.equals(expected_jobs)


def test_job_queue_add():
    job_queue = _JobQueue()
    job_queue.add(5, 'open')
    assert job_queue.jobs.equals(
        pd.Series(['open', 'open', 'open', 'open', 'open'], dtype='string'))


def test_job_queue_reset(sample_job_queue: _JobQueue):
    sample_job_queue.reset()
    assert sample_job_queue.jobs.empty


def test_job_queue_get_open_job(sample_job_queue: _JobQueue):
    open_job_index = sample_job_queue.get_open_job()
    assert isinstance(open_job_index, int)
    assert open_job_index in sample_job_queue.jobs.index


def test_job_queue_get_open_job_no_jobs():
    jobs = pd.Series(['finished', 'finished', 'in progress',
                     'finished'], dtype='string')
    job_queue = _JobQueue(jobs)
    with pytest.raises(NoOpenJobsError):
        job_queue.get_open_job()


def test_job_queue_mark_as_in_progress(sample_job_queue: _JobQueue):
    # sample_job_queue.mark_as_in_progress(0)
    sample_job_queue.mark(0, Status.IN_PROGRESS)
    assert sample_job_queue.jobs.loc[0] == 'in progress'


def test_job_queue_mark_as_finished(sample_job_queue: _JobQueue):
    # sample_job_queue.mark_as_finished(1)
    sample_job_queue.mark(1, Status.FINISHED)
    assert sample_job_queue.jobs.loc[1] == 'finished'


def test_job_queue_mark_as_error(sample_job_queue: _JobQueue):
    # sample_job_queue.mark_as_error(2)
    sample_job_queue.mark(2, Status.ERROR)
    assert sample_job_queue.jobs.loc[2] == 'error'


def test_job_queue_mark_all_in_progress_open(sample_job_queue: _JobQueue):
    sample_job_queue.mark_all_in_progress_open()
    assert sample_job_queue.jobs.equals(
        pd.Series(['open', 'open', 'open', 'finished'], dtype='string'))


def test_job_queue_is_all_finished(sample_job_queue: _JobQueue):
    assert not sample_job_queue.is_all_finished()
