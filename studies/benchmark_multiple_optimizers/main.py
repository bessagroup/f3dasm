import logging
from typing import List
import autograd.numpy as np
import hydra
from hydra.core.config_store import ConfigStore
import pickle
import f3dasm
from config import Config


def convert_config_to_input(config: Config) -> List[dict]:

    seed = np.random.randint(low=0, high=1e5)

    function_class: f3dasm.Function = f3dasm.find_class(f3dasm.functions, config.function.function_name)
    optimizers_class: List[f3dasm.Optimizer] = [
        f3dasm.find_class(f3dasm.optimization, optimizer_name) for optimizer_name in config.optimizers.optimizers_names
    ]
    sampler_class: f3dasm.Sampler = f3dasm.find_class(f3dasm.sampling, config.sampler.sampler_name)

    bounds = np.tile([config.design.lower_bound, config.design.upper_bound], (config.design.dimensionality, 1))
    design = f3dasm.make_nd_continuous_design(bounds=bounds, dimensionality=config.design.dimensionality)
    data = f3dasm.ExperimentData(design=design)

    function = function_class(
        dimensionality=config.design.dimensionality, noise=config.function.noise, scale_bounds=bounds, seed=seed
    )
    optimizers = [optimizer(data=data, seed=seed) for optimizer in optimizers_class]
    sampler = sampler_class(design=data.design, seed=seed)

    return [
        {
            "optimizer": optimizer,
            "function": function,
            "sampler": sampler,
            "number_of_samples": config.sampler.number_of_samples,
            "iterations": config.execution.iterations,
            "realizations": config.execution.realizations,
            "parallelization": config.execution.parallelization,
            "seed": seed,
        }
        for optimizer in optimizers
    ]


@hydra.main(config_path=".", config_name="config")
def main(cfg: Config):
    options_list = convert_config_to_input(config=cfg)

    results = []

    for options in options_list:
        results.append(f3dasm.run_multiple_realizations(**options))

    f3dasm.write_pickle("0006", results)


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()