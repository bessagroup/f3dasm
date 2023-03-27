import logging
from typing import List
import autograd.numpy as np
import hydra
from hydra.core.config_store import ConfigStore
import pickle
import f3dasm
from config import Config
import gpytorch
import torch
from sklearn.preprocessing import StandardScaler
from fidelity_augmentors import Warp, Scale, NoiseInterpolator
from generate_train_data import *

from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)


def convert_config_to_input(config: Config) -> List[dict]:

    seed = np.random.randint(low=0, high=1e5)
    # seed = 123

    with open('seed.txt', "w") as f:
        f.write(str(seed))

    bounds = np.tile([0.0, 1.0], (config.design.dimensionality, 1))
    design = f3dasm.make_nd_continuous_design(bounds=bounds, dimensionality=config.design.dimensionality)

    regressor_class = f3dasm.find_class(f3dasm.machinelearning.gpr, config.regressor_name)

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

    param_class = f3dasm.find_class(f3dasm.machinelearning.gpr, config.regressor_name + '_Parameters')

    if config.regressor_name == 'Sogpr':
        kernel = kernel[0]
        mean = mean[0]

    param = param_class(
        kernel=kernel,
        mean=mean,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        noise_fix=True,#not config.aug_type == 'noise',
        opt_algo=torch.optim.Adam,
        opt_algo_kwargs=dict(lr=0.1),
        verbose_training=False,
        training_iter=50,
        )

    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

    fun_class: f3dasm.Function = f3dasm.find_class(f3dasm.functions, config.function_name)

    fun = fun_class(dimensionality=config.design.dimensionality, scale_bounds=bounds, seed=seed, offset=False)

    if config.regressor_name == 'Sogpr':
        train_data, scaler = train_data_single_fidelity(config=config, sampler=sampler, fun=fun)
    
    else:
        train_data, scaler = train_data_multi_fidelity_noise(config=config, sampler=sampler, fun=fun)
    
    ## Define regressor
    regressor = regressor_class(
        train_data=train_data,
        parameter=param,
    )

    surrogate = regressor.train()

    n_test = 500

    ## Get into evaluation (predictive posterior) mode
    surrogate.model.eval()
    surrogate.model.likelihood.eval()

    ## Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_sampler = f3dasm.sampling.SobolSequence(design=design, seed=0)

        test_x = torch.tensor(test_sampler.get_samples(numsamples=n_test).get_input_data().values)

        if config.design.dimensionality == 1:
            sort_indices = test_x.argsort(axis=0)
            test_x = test_x[sort_indices].squeeze(axis=-1)

        # test_x = torch.linspace(0, 1, n_test)

        if config.regressor_name == 'Cokgj':
            test_x = [torch.empty(0, config.design.dimensionality), test_x.clone()]
        
        observed_pred = surrogate.predict(test_x)

        if config.regressor_name == 'Cokgj':
            test_x_plot = test_x[-1]
            train_x = torch.tensor(train_data[-1].get_input_data().values)
            train_y = torch.tensor(scaler.inverse_transform(train_data[-1].get_output_data().values))

        else:
            test_x_plot = test_x
            train_x = torch.tensor(train_data.get_input_data().values)
            train_y = torch.tensor(scaler.inverse_transform(train_data.get_output_data().values))
        
        exact_y = fun(test_x_plot)

        if config.design.dimensionality == 1:
            surrogate.plot_gpr(
                test_x=test_x_plot.flatten(),
                scaler=scaler,
                exact_y=exact_y,
                observed_pred=observed_pred,
                train_x=train_x,
                train_y=train_y,
                savefig=False,
            )

    options = {
        "function": fun,
        "surrogate": surrogate,
        "scaler": scaler,
        "observed_pred": observed_pred,
        "exact_y": exact_y,
        }

    return options

@hydra.main(config_path=".", config_name="default")
def main(cfg: Config):
    options = convert_config_to_input(config=cfg)
    
    ## Calculate and store metrics
    metrics_df = options['surrogate'].gp_metrics(
        scaler=options['scaler'],
        observed_pred=options['observed_pred'],
        exact_y=options['exact_y'].flatten(),
    )

    metrics_df.to_csv(options['function'].name + '.csv')

cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
