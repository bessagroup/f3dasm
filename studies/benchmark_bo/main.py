import logging
from typing import List
import autograd.numpy as np
import hydra
from hydra.core.config_store import ConfigStore
import pickle

from matplotlib import pyplot as plt
import f3dasm
from config import Config
import itertools
import copy
import gpytorch
import torch
from f3dasm.functions.fidelity_augmentors import NoiseInterpolator, NoisyBaseFun, Scale, BiasedBaseFun

from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)

def scale_factor(input_row: np.ndarray) -> np.ndarray:
    res = input_row + .5
    return res

def convert_config_to_input(config: Config) -> List[dict]:

    if config.execution.seed == -1:
        seed = np.random.randint(low=0, high=1e5)
    else:
        seed = config.execution.seed

    log.info("RANDOM SEED = %d" % seed)

    bounds = np.tile([0.0, 1.0], (config.dimensionality, 1))
    design = f3dasm.make_nd_continuous_design(bounds=bounds, dimensionality=config.dimensionality)

    fun_class: f3dasm.Function = f3dasm.find_class(f3dasm.functions, config.function_name)
    fun = fun_class(dimensionality=config.dimensionality, scale_bounds=bounds, seed=seed, offset=False)

    sampler_class: f3dasm.Sampler = f3dasm.find_class(f3dasm.sampling, config.sampler.sampler_name)
    sampler = sampler_class(design=design, seed=seed)

    kernel = torch.nn.ModuleList([
        gpytorch.kernels.ScaleKernel(f3dasm.find_class(gpytorch.kernels, config.regressor.kernel_name)()),
        gpytorch.kernels.ScaleKernel(f3dasm.find_class(gpytorch.kernels, config.regressor.kernel_name)()),
    ])

    mean = torch.nn.ModuleList([
        f3dasm.find_class(gpytorch.means, config.regressor.mean_name)(),
        f3dasm.find_class(gpytorch.means, config.regressor.mean_name)(),
    ])


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
        noise_fix=False,#not config.aug_type == 'noise',
        training_iter=150,
        verbose_training=False,
        opt_algo_kwargs=dict(lr=0.1)
        )
    
    if config.data_fidelity_structure == 'single':
        iterations = config.budget
        objective_function = fun
        sampler = f3dasm.sampling.SobolSequence_torch(design=design, seed=seed)

    else:
        iterations = config.execution.iterations
        fidelity_functions = []
        multifidelity_samplers = []
        sampler = []

        for i, aug_coeff in enumerate(config.aug_coeffs):

            noise_augmentor = NoiseInterpolator(noise_var=1., aug_coeff=aug_coeff)
            # scale_augmentor = Scale(scale=scale, reverse_scale=reverse_scale)

            if i == 0:
                fidelity_function = NoisyBaseFun(
                    seed=seed,
                    base_fun=fun, 
                    dimensionality=config.dimensionality,
                    noise_augmentor=noise_augmentor, 
                    )                
                
                # fidelity_function = BiasedBaseFun(
                #     seed=seed,
                #     base_fun=fun, 
                #     dimensionality=config.dimensionality,
                #     scale_augmentor=scale_augmentor, 
                #     )

            else:
                fidelity_function = fun

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
        visualize_gp=False,
    )

    optimizer_class: f3dasm.Optimizer = f3dasm.find_class(f3dasm.optimization, optimizer_name)
    optimizer = optimizer_class(data=f3dasm.ExperimentData(design=design), seed=seed)
    optimizer.parameter = optimizer_parameters

    options = {
        "optimizer": optimizer,
        "function": objective_function,
        "sampler": sampler,
        "iterations": iterations,
        "seed": seed,
        "number_of_samples": number_of_samples,
        }
    
    if config.data_fidelity_structure == 'multi':
        options.update(dict(budget=config.budget))

    return options, config.data_fidelity_structure


@hydra.main(config_path=".", config_name="default")
def main(cfg: Config):
    options, data_fidelity_structure = convert_config_to_input(config=cfg)

    if data_fidelity_structure == 'single':
        result = f3dasm.run_optimization(**options)
        result.data.to_csv(options['function'].name + '.csv')
    else:
        result = f3dasm.run_multi_fidelity_optimization(**options)
        result[0].data.to_csv(options['function'].fidelity_functions[-1].name + '_lf.csv')
        result[-1].data.to_csv(options['function'].fidelity_functions[-1].name + '_hf.csv')

        plt.show()


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
