import logging
import autograd.numpy as np
import hydra
from hydra.core.config_store import ConfigStore

import f3dasm


def convert_config_to_input(config: f3dasm.Config) -> dict:

    function_class: f3dasm.Function = f3dasm.find_class(f3dasm.functions, config.function.function_name)
    optimizer_class: f3dasm.Optimizer = f3dasm.find_class(f3dasm.optimization, config.optimizer.optimizer_name)
    sampler_class: f3dasm.SamplingInterface = f3dasm.find_class(f3dasm.sampling, config.sampler.sampler_name)

    bounds = np.tile([config.design.lower_bound, config.design.upper_bound], (config.design.dimensionality, 1))
    design = f3dasm.make_nd_continuous_design(bounds=bounds, dimensionality=config.design.dimensionality)
    data = f3dasm.Data(design=design)

    function = function_class(dimensionality=config.design.dimensionality, noise=config.function.noise)
    optimizer = optimizer_class(data=data, hyperparameters=config.optimizer.hyperparameters)
    sampler = sampler_class(design=data.design)

    return {
        "optimizer": optimizer,
        "function": function,
        "sampler": sampler,
        "iterations": config.execution.iterations,
        "realizations": config.execution.realizations,
        "parallelization": config.execution.parallelization,
    }


@hydra.main(config_path=".", config_name="config")
def main(cfg: f3dasm.Config):
    log.info("testmessage")
    # print(cfg)
    options = convert_config_to_input(config=cfg)
    results = f3dasm.run_multiple_realizations(**options)
    print(results)
    print(cfg)


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=f3dasm.Config)

log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
