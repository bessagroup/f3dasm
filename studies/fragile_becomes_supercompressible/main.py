

"""
Main entrypoint of the experiment

Functions
---------

main
    Main script to call
pre_processing
    Pre-processing steps
process
    Main process to execute
"""

#
#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path
from time import sleep
from typing import Optional

# Third-party
import hydra
import numpy as np
import pandas as pd
from f3dasm import ExperimentData
from f3dasm import logger as f3dasm_logger
from f3dasm.design import Domain

from abaqus2py import F3DASMAbaqusSimulator

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


#                                                         Custom sampler method
# =============================================================================

def log_normal_sampler(domain: Domain, n_samples: int,
                       mean: float, sigma: float, seed: Optional[int] = None):
    """Sampler function for lognormal distribution

    Parameters
    ----------
    domain
        Domain object
    n_samples
        Number of samples to generate
    mean
        Mean of the lognormal distribution
    sigma
        Standard deviation of the lognormal distribution
    seed
        Seed for the random number generator

    Returns
    -------
    DataFrame
        pandas DataFrame with the samples
    """
    rng = np.random.default_rng(seed)
    sampled_imperfections = rng.lognormal(
        mean=mean, sigma=sigma, size=n_samples)
    return pd.DataFrame(sampled_imperfections, columns=domain.names)

# =============================================================================


def pre_processing(config):
    experimentdata = ExperimentData.from_yaml(config.experimentdata)

    if 'from_sampling' in config.imperfection:
        domain_imperfections = Domain.from_yaml(config.imperfection.domain)

        imperfections = ExperimentData.from_sampling(
            sampler=log_normal_sampler,
            domain=domain_imperfections,
            n_samples=config.experimentdata.from_sampling.n_samples,
            mean=config.imperfection.mean,
            sigma=config.imperfection.sigma,
            seed=config.experimentdata.from_sampling.seed)

        experimentdata = experimentdata.join(imperfections)

    experimentdata.store(Path.cwd())

    # Create directories for ABAQUS results
    (Path.cwd() / 'lin_buckle').mkdir(exist_ok=True)
    (Path.cwd() / 'riks').mkdir(exist_ok=True)


def post_processing(config):
    ...


def process(config):
    """Main script that handles the execution of open jobs

    Parameters
    ----------
    config
        Hydra configuration file object
    """
    # if 'from_file' in config.experimentdata:
    #     project_dir = config.experimentdata.from_file

    # else:
    project_dir = Path().cwd()

    # Retrieve the ExperimentData object
    max_tries = 500
    tries = 0

    while tries < max_tries:
        try:
            data = ExperimentData.from_file(project_dir)
            break  # Break out of the loop if successful
        except FileNotFoundError:
            tries += 1
            sleep(10)

    if tries == max_tries:
        raise FileNotFoundError(f"Could not open ExperimentData after "
                                f"{max_tries} attempts.")

    simulator_lin_buckle = F3DASMAbaqusSimulator(
        py_file=config.scripts.lin_buckle_pre,
        post_py_file=config.scripts.lin_buckle_post,
        working_directory=Path.cwd() / 'lin_buckle',
        max_waiting_time=60)
    simulator_riks = F3DASMAbaqusSimulator(
        py_file=config.scripts.riks_pre,
        post_py_file=config.scripts.riks_post,
        working_directory=Path.cwd() / 'riks',
        max_waiting_time=120)

    data.evaluate(data_generator=simulator_lin_buckle, mode=config.mode)

    data.store()

    data.mark_all('open')

    data.evaluate(data_generator=simulator_riks, mode=config.mode)

    if config.mode == 'sequential':
        # Store the ExperimentData to a csv file
        data.store()


@hydra.main(config_path=".", config_name="config")
def main(config):
    """Main script to call

    Parameters
    ----------
    config
        Configuration parameters defined in config.yaml
    """

    f3dasm_logger.setLevel(config.log_level)
    # Execute the initial_script for the first job
    if config.hpc.jobid == 0:
        pre_processing(config)
        post_processing(config)

    elif config.hpc.jobid == -1:  # Sequential
        pre_processing(config)
        process(config)
        post_processing(config)

    else:
        sleep(3*config.hpc.jobid)  # To asynchronize the jobs
        process(config)


if __name__ == "__main__":
    main()
