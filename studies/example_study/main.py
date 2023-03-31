import os
from time import sleep

import hydra
from config import Config  # NOQA
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from myscript import main_parallel

import f3dasm


def initial_script(config: Config):

    SEED = config.experiment.seed
    NUMSAMPLES = config.design.number_of_samples

    # Create the designspace
    param_a = f3dasm.ContinuousParameter(name='a', lower_bound=-1., upper_bound=1.)
    param_b = f3dasm.ContinuousParameter(name='b', lower_bound=-1., upper_bound=1.)
    param_c = f3dasm.ContinuousParameter(name='c', lower_bound=-1., upper_bound=1.)
    param_out = f3dasm.ContinuousParameter(name='y')

    design = f3dasm.DesignSpace(input_space=[param_a, param_b, param_c], output_space=[param_out])

    # Create the sampler object
    sampler = f3dasm.sampling.RandomUniform(design=design, seed=SEED)

    # Create the experimentdata object
    data: f3dasm.ExperimentData = sampler.get_samples(numsamples=NUMSAMPLES)

    # Write the experimentdata to a file
    data.store(filename=config.file.experimentdata_filename)

    # Create the JobQueue object
    job_queue = f3dasm.experiment.JobQueue(filename=config.file.jobqueue_filename)
    job_queue.create_jobs_from_experimentdata(data)

    # Write the JobQueue object to a file
    job_queue.write_new_jobfile()


def job(config: Config):

    # Retrieve the queue
    job_queue = f3dasm.experiment.JobQueue(filename=config.file.jobqueue_filename)

    while True:
        try:
            jobnumber = job_queue.get()
        except f3dasm.experiment.NoOpenJobsError:
            break

        data = f3dasm.design.load_experimentdata(filename=config.file.experimentdata_filename)
        args = data.get_inputdata_by_index(jobnumber)

        value = main_parallel(**data.get_inputdata_by_index(jobnumber))
        data.set_outputdata_by_index(jobnumber, value)

        data.store(filename=config.file.experimentdata_filename)

        job_queue.mark_finished(jobnumber)


@hydra.main(config_path=".", config_name="config")
def main(config: Config):
    """Main script to call

    Parameters
    ----------
    config
        Configuration parameters defined in config.yaml
    """
    # Execute the initial_script for the first job
    if config.hpc.jobid == 0:
        initial_script(config)

    elif config.hpc.jobid == -1:  # Sequential
        initial_script(config)
        job(config)

    else:
        sleep(config.hpc.jobid)  # Just to asynchronize the jobs
        job(config)


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

if __name__ == "__main__":
    main()
