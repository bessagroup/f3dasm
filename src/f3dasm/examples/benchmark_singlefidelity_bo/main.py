import logging
from typing import List
import autograd.numpy as np
import hydra
from hydra.core.config_store import ConfigStore
import pickle
import f3dasm
from config import Config
import itertools
import copy
import gpytorch


def convert_config_to_input(config: Config) -> List[dict]:

    seed = np.random.randint(low=0, high=1e5)

    function_class: List[f3dasm.Function] = [
        f3dasm.find_class(f3dasm.functions, function_name) for function_name in config.functions.function_names
    ]

    optimizer_class: f3dasm.Optimizer = f3dasm.find_class(f3dasm.optimization, config.optimizer.optimizer_name)

    sampler_class: f3dasm.SamplingInterface = f3dasm.find_class(f3dasm.sampling, config.sampler.sampler_name)

    bounds = np.tile([config.design.lower_bound, config.design.upper_bound], (config.design.dimensionality, 1))
    design = f3dasm.make_nd_continuous_design(bounds=bounds, dimensionality=config.design.dimensionality)
    data = f3dasm.Data(design=design)

    functions = [
        f_class(dimensionality=config.design.dimensionality, noise=config.functions.noise, scale_bounds=bounds, seed=seed)
        for f_class in function_class
    ]
    optimizer = optimizer_class(data=data, seed=seed)

    optimizer.init_parameters()
    optimizer.parameter.kernel = gpytorch.kernels.ScaleKernel(base_kernel=gpytorch.kernels.RBFKernel())
    optimizer.parameter.noise_fix = False

    sampler = sampler_class(design=data.design, seed=seed)

    return [
        {
            "optimizer": optimizer,
            "function": f,
            "sampler": sampler,
            "number_of_samples": config.sampler.number_of_samples,
            "iterations": config.execution.iterations,
            "seed": seed,
        }
        for f in functions
    ]


@hydra.main(config_path=".", config_name="default")
def main(cfg: Config):
    options_list = convert_config_to_input(config=cfg)

    results = []

    for options in options_list:

        # options['optimizer'].init_parameters()
        # options['optimizer'].parameter.kernel = gpytorch.kernels.ScaleKernel(base_kernel=gpytorch.kernels.RBFKernel())
        # options['optimizer'].parameter.noise_fix = False

        result = f3dasm.run_optimization(**options)
        result.data.to_csv(options['function'].name + '.csv')

        results.append(result)

    print([result.data for result in results])

cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
