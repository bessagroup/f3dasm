import logging
from typing import List

import hydra
import numpy as np
from config import Config
from hydra.core.config_store import ConfigStore

import f3dasm


def convert_config_to_input(config: Config) -> List[dict]:

    seed = np.random.randint(low=0, high=1e5)
    dimensionality = np.random.randint(
        low=config.design.dimensionality_lower_bound, high=config.design.dimensionality_upper_bound + 1
    )

    functions_list = f3dasm.functions.get_functions(d=dimensionality)

    function_class = np.random.choice(functions_list)

    function_noise = float(np.random.choice([config.function.noise, 0.0]))

    optimizers_class: List[f3dasm.Optimizer] = [
        f3dasm.find_class(f3dasm.optimization, optimizer_name) for optimizer_name in config.optimizers.optimizers_names
    ]
    sampler_class: f3dasm.Sampler = f3dasm.find_class(f3dasm.sampling, config.sampler.sampler_name)

    bounds = np.tile([config.design.lower_bound, config.design.upper_bound], (dimensionality, 1))

    design = f3dasm.make_nd_continuous_design(bounds=bounds, dimensionality=dimensionality)
    function = function_class(
        dimensionality=dimensionality, noise=function_noise, scale_bounds=design.get_bounds(), seed=seed
    )
    data = f3dasm.Data(design=design)

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

    for _ in range(cfg.execution.number_of_functions):
        options_list = convert_config_to_input(config=cfg)

        results = []

        for options in options_list:
            results.append(f3dasm.run_multiple_realizations(**options))

        filename: str = f"{options_list[0]['function'].get_name()}_seed{options_list[0]['seed']}\
            _dim{options_list[0]['function'].dimensionality}"
        f3dasm.write_pickle(filename, results)

        df = f3dasm.margin_of_victory(results)

        f3dasm.write_pickle('mov_'+filename, df)


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
