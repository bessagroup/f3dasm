

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

# Third-party
import hydra
import numpy as np
import pandas as pd
import xarray as xr

# Local
from f3dasm import ExperimentData
from f3dasm.datageneration import DataGenerator
from f3dasm.datageneration.functions import get_functions
from f3dasm.design import Domain, Sampler, make_nd_continuous_domain

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

class CustomSampler(Sampler):
    def __init__(self, seed: int):
        self.seed = seed

    def sample(self, n_samples: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        samples = []

        for i in range(n_samples):
            dim = rng.choice(self.domain.space['dimensionality'].categories)

            available_functions = list(set(get_functions(d=int(dim))) & set(
                self.domain.space['function_name'].categories))
            function_name = rng.choice(available_functions)

            noise = rng.choice(self.domain.space['noise'].categories)
            seed = rng.integers(
                low=self.domain.space['seed'].lower_bound,
                high=self.domain.space['seed'].upper_bound)
            budget = self.domain.space['budget'].value

            samples.append([function_name, dim, noise, seed, budget])

        return pd.DataFrame(
            samples, columns=self.domain.names)[self.domain.names]

#                                                          Custom datagenerator
# =============================================================================


class BenchmarkOptimizer(DataGenerator):
    def __init__(self, config):
        self.config = config

    def optimize_function(self, optimizer: dict) -> xr.Dataset:
        seed = self.experiment_sample.get('seed')
        function_name = self.experiment_sample.get('function_name')
        dimensionality = self.experiment_sample.get('dimensionality')
        noise = self.experiment_sample.get('noise')
        budget = self.experiment_sample.get('budget')

        hyperparameters = optimizer['hyperparameters'] \
            if 'hyperparameters' in optimizer else {}

        # inside loop
        data_list = []
        for r in range(self.config.optimization.realizations):

            domain = make_nd_continuous_domain(
                bounds=np.tile(
                    [self.config.optimization.lower_bound,
                     self.config.optimization.upper_bound],
                    (dimensionality, 1)))
            data = ExperimentData.from_sampling(
                sampler=self.config.optimization.sampler_name, domain=domain,
                n_samples=self.config.optimization.number_of_samples,
                seed=seed + r)

            data.evaluate(
                data_generator=function_name,
                scale_bounds=domain.get_bounds(), offset=True, noise=noise,
                seed=seed, mode='sequential')

            data.optimize(
                optimizer=optimizer['name'], data_generator=function_name,
                kwargs={'scale_bounds': domain.get_bounds(
                ), 'offset': True, 'noise': noise, 'seed': seed},
                iterations=budget, x0_selection='best',
                hyperparameters={'seed': seed + r,
                                 **hyperparameters})

            data_list.append(data.to_xarray())

        return xr.concat(data_list, dim=xr.DataArray(
            range(self.config.optimization.realizations), dims='realization'))

    def execute(self):
        for optimizer in self.config.optimization.optimizers:
            opt_results = self.optimize_function(optimizer)

            self.experiment_sample.store(
                object=opt_results, name=optimizer['name'], to_disk=True)

#                                                          Data-driven workflow
# =============================================================================


def pre_processing(config):

    custom_sampler = CustomSampler(
        seed=config.experimentdata.from_sampling.seed)

    if 'from_sampling' in config.experimentdata:
        experimentdata = ExperimentData.from_sampling(
            sampler=custom_sampler,
            domain=Domain.from_yaml(config.domain),
            n_samples=config.experimentdata.from_sampling.n_samples,
        )

    else:
        experimentdata = ExperimentData.from_yaml(config.experimentdata)

    experimentdata.store(Path.cwd())


def process(config):
    """Main script that handles the execution of open jobs

    Parameters
    ----------
    config
        Hydra configuration file object
    """
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

    benchmark_optimizer = BenchmarkOptimizer(config)

    data.evaluate(data_generator=benchmark_optimizer, mode=config.mode)

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
    # Execute the initial_script for the first job
    if config.hpc.jobid == 0:
        pre_processing(config)

    elif config.hpc.jobid == -1:  # Sequential
        pre_processing(config)
        process(config)

    else:
        sleep(3*config.hpc.jobid)  # To asynchronize the jobs
        process(config)


if __name__ == "__main__":
    main()
