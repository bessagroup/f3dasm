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
import torch
from fidelity_augmentors import NoiseInterpolator

from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)

class FidelityFunction(f3dasm.functions.Function):
    def __init__(self, seed=None, fidelity_augmentor=None, fun=None, dimensionality=None,):
        super().__init__(seed)
        self.fidelity_augmentor = fidelity_augmentor
        self.fun = fun
        self.scale_bounds = fun.scale_bounds
        self.dimensionality = dimensionality
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        res = self.fidelity_augmentor.augment(self.fun(x))
        return res

def convert_config_to_input(config: Config) -> List[dict]:

    seed = np.random.randint(low=0, high=1e5)

    bounds = np.tile([0.0, 1.0], (config.dimensionality, 1))
    design = f3dasm.make_nd_continuous_design(bounds=bounds, dimensionality=config.dimensionality)

    fun_class: f3dasm.Function = f3dasm.find_class(f3dasm.functions, config.function_name)
    fun = fun_class(dimensionality=config.dimensionality, scale_bounds=bounds, seed=seed, offset=False)

    sampler_class: f3dasm.Sampler = f3dasm.find_class(f3dasm.sampling, config.sampler.sampler_name)
    sampler = sampler_class(design=design, seed=seed)

    kernel = [
        gpytorch.kernels.ScaleKernel(f3dasm.find_class(gpytorch.kernels, config.regressor.kernel_name)()),
        gpytorch.kernels.ScaleKernel(f3dasm.find_class(gpytorch.kernels, config.regressor.kernel_name)()),
    ]

    mean = [
        f3dasm.find_class(gpytorch.means, config.regressor.mean_name)(),
        f3dasm.find_class(gpytorch.means, config.regressor.mean_name)(),
    ]


    if config.data_fidelity_structure == 'single':
        kernel = kernel[0]
        mean = mean[0]
        number_of_samples = config.sampler.number_of_samples
        regressor_name = 'Sogpr'
        acquisition_name = 'UpperConfidenceBound'
        optimizer_name = 'BayesianOptimizationTorch'
    
    else:
        number_of_samples = [int(np.ceil(config.sampler.number_of_lf_samples)), 
                             int(np.ceil(config.sampler.number_of_hf_samples))]
        regressor_name = 'Cokgj'
        acquisition_name = 'VFUpperConfidenceBound'
        optimizer_name = 'MFBayesianOptimizationTorch'

    regressor_class = f3dasm.find_class(f3dasm.machinelearning.gpr, regressor_name)

    regressor_parameters_class = f3dasm.find_class(f3dasm.machinelearning.gpr, regressor_name + '_Parameters')
    regressor_parameters = regressor_parameters_class(
        kernel=kernel,
        mean=mean,
        noise_fix=not config.aug_type == 'noise',
        training_iter=150,
        )
    
    if config.data_fidelity_structure == 'single':
        objective_function = fun
        sampler = f3dasm.sampling.SobolSequence_torch(design=design, seed=seed)

    else:
        fidelity_functions = []
        multifidelity_samplers = []
        sampler = []

        for i, aug_coeff in enumerate(config.aug_coeffs):

            noise_interpolator = NoiseInterpolator(noise_var=1., aug_coeff=aug_coeff)

            # if i == 0:
            #     fidelity_function = lambda x: noise_interpolator.augment(fun(x))
            # else:
            #     fidelity_function = fu

            # fidelity_function = lambda x: noise_interpolator.augment(fun(x))
            fidelity_function = FidelityFunction(
                fidelity_augmentor=noise_interpolator, 
                fun=fun, 
                dimensionality=config.dimensionality,
                )

            sampler_fid = f3dasm.sampling.SobolSequence_torch(design=design, seed=seed)

            fidelity_functions.append(fidelity_function)
            multifidelity_samplers.append(sampler)
            sampler.append(sampler_fid)

        multifidelity_function = f3dasm.functions.MultiFidelityFunction(
            fidelity_functions=fidelity_functions,
            fidelity_parameters=config.aug_coeffs,
            costs=config.costs,
        )

        objective_function = multifidelity_function
    
    acquisition_class = f3dasm.find_class(f3dasm.machinelearning.acquisition_functions, acquisition_name)

    optimizer_parameters_class = f3dasm.find_class(f3dasm.optimization, optimizer_name + '_Parameters')
    optimizer_parameters = optimizer_parameters_class(
        regressor=regressor_class,
        regressor_hyperparameters=regressor_parameters,
        acquisition=acquisition_class,
        acquisition_hyperparameters=f3dasm.optimization.Acquisition_Parameters(),
        n_init=config.sampler.number_of_samples,
    )

    optimizer_class: f3dasm.Optimizer = f3dasm.find_class(f3dasm.optimization, optimizer_name)
    optimizer = optimizer_class(data=f3dasm.ExperimentData(design=design), seed=seed)
    optimizer.parameter = optimizer_parameters

    options = {
        "optimizer": optimizer,
        "function": objective_function,
        "sampler": sampler,
        "iterations": config.execution.iterations,
        "seed": seed,
        "number_of_samples": number_of_samples,
        }
    
    if config.data_fidelity_structure == 'multi':
        options.update(dict(budget=config.budget))

    return options, config.data_fidelity_structure


@hydra.main(config_path=".", config_name="default2")
def main(cfg: Config):
    options, data_fidelity_structure = convert_config_to_input(config=cfg)

    if data_fidelity_structure == 'single':
        result = f3dasm.run_optimization(**options)
        result.data.to_csv(options['function'].name + '.csv')
    else:
        result = f3dasm.run_multi_fidelity_optimization(**options)
        result[-1].data.to_csv(options['function'].fidelity_functions[-1].fun.name + '.csv')


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
