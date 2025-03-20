

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
from f3dasm import (Block, ExperimentData, ExperimentSample,
                    create_datagenerator, create_optimizer, create_sampler)
from f3dasm.datageneration import DataGenerator
from f3dasm.datageneration.functions import get_functions
from f3dasm.design import Domain, make_nd_continuous_domain

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

class CustomSampler(Block):
    def __init__(self, seed: int):
        self.seed = seed

    def call(self, data: ExperimentData, n_samples: int) -> ExperimentData:
        rng = np.random.default_rng(self.seed)
        samples = []

        for i in range(n_samples):
            dim = rng.choice(
                data._domain.input_space['dimensionality'].categories)

            available_functions = list(set(get_functions(d=int(dim))) & set(
                data._domain.input_space['function_name'].categories))
            function_name = rng.choice(available_functions)

            noise = rng.choice(data._domain.input_space['noise'].categories)
            seed = rng.integers(
                low=data._domain.input_space['seed'].lower_bound,
                high=data._domain.input_space['seed'].upper_bound)
            budget = data._domain.input_space['budget'].value

            samples.append([function_name, dim, noise, seed, budget])

        df = pd.DataFrame(
            samples,
            columns=data._domain.input_names)[data._domain.input_names]

        return ExperimentData(
            domain=data._domain, input_data=df,
            project_dir=data.project_dir)

#                                                          Custom datagenerator
# =============================================================================


class BenchmarkOptimizer(DataGenerator):
    def __init__(self, config):
        self.config = config

    def optimize_function(self, experiment_sample: ExperimentSample,
                          optimizer: dict) -> xr.Dataset:
        seed = experiment_sample.input_data['seed']
        function_name = experiment_sample.input_data['function_name']
        dimensionality = experiment_sample.input_data['dimensionality']
        noise = experiment_sample.input_data['noise']
        budget = experiment_sample.input_data['budget']

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
            sampler = create_sampler(
                sampler=self.config.optimization.sampler_name,
                seed=seed + r)

            data = ExperimentData(domain=domain)

            sampler.arm(data=data)
            data = sampler.call(
                data=data,
                n_samples=self.config.optimization.number_of_samples)

            data_generator = create_datagenerator(
                data_generator=function_name,
                scale_bounds=domain.get_bounds(),
                offset=True, noise=noise, seed=seed + r)

            data_generator.arm(data=data)

            data = data_generator.call(data=data, mode='sequential')

            _optimizer = create_optimizer(optimizer=optimizer['name'],
                                          seed=seed + r, **hyperparameters)

            data.optimize(
                optimizer=_optimizer, data_generator=data_generator,
                iterations=budget, x0_selection='best',
            )

            data_list.append(data.to_xarray())

        return xr.concat(data_list, dim=xr.DataArray(
            range(self.config.optimization.realizations), dims='realization'))

    def execute(self, experiment_sample: ExperimentSample) -> ExperimentSample:
        for optimizer in self.config.optimization.optimizers:
            opt_results = self.optimize_function(
                experiment_sample=experiment_sample,
                optimizer=optimizer)

            experiment_sample.store(
                object=opt_results, name=optimizer['name'], to_disk=True)

        return experiment_sample

#                                                          Data-driven workflow
# =============================================================================


def pre_processing(config):

    custom_sampler = CustomSampler(
        seed=config.experimentdata.from_sampling.seed)

    if 'from_sampling' in config.experimentdata:
        experimentdata = ExperimentData(domain=Domain.from_yaml(config.domain))

        custom_sampler.arm(data=experimentdata)

        experimentdata = custom_sampler.call(
            data=experimentdata,
            n_samples=config.experimentdata.from_sampling.n_samples
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

    benchmark_optimizer.arm(data=data)

    data = benchmark_optimizer.call(data=data, mode=config.mode)

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
