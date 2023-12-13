import pandas as pd

from f3dasm._src.experimentdata._jobqueue import _JobQueue


def test_select_all_with_matching_status():
    # Create a job queue with some jobs
    job_queue = _JobQueue()
    job_queue.jobs = pd.Series(['in progress', 'running', 'completed', 'in progress', 'failed'])

    # Select all jobs with status 'in progress'
    selected_jobs = job_queue.select_all('in progress')

    # Check if the selected jobs match the expected result
    assert (selected_jobs.jobs == ['in progress', 'in progress']).all()

def test_select_all_with_no_matching_status():
    # Create a job queue with some jobs
    job_queue = _JobQueue()
    job_queue.jobs = pd.Series(['in progress', 'running', 'completed', 'in progress', 'failed'])

    # Select all jobs with status 'cancelled'
    selected_jobs = job_queue.select_all('cancelled')

    # Check if the selected jobs match the expected result
    assert selected_jobs.jobs.empty

def test_select_all_with_empty_job_queue():
    # Create an empty job queue
    job_queue = _JobQueue()

    # Select all jobs with status 'in progress'
    selected_jobs = job_queue.select_all('in progress')

    # Check if the selected jobs match the expected result
    assert selected_jobs.jobs.empty
