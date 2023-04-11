import logging
from typing import List
import autograd.numpy as np
import hydra
from hydra.core.config_store import ConfigStore
import pickle

from matplotlib import pyplot as plt
import f3dasm
from config import Config
import gpytorch
import torch
from sklearn.preprocessing import StandardScaler
from generate_train_data import *

from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)


def convert_config_to_input(config: Config) -> List[dict]:

    if config.seed == -1:
        seed = np.random.randint(low=0, high=1e5)
    else:
        seed = config.seed

    log.info("RANDOM SEED = %d" % seed)

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

    if config.regressor_name in ['Sogpr', 'MultitaskGPR']:
        kernel = kernel[0]
        mean = mean[0]

    noise_fix = not config.aug_type == 'noise' or config.regressor_name == 'Sogpr'

    param = param_class(
        kernel=kernel,
        mean=mean,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        noise_fix=noise_fix,
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

        if config.regressor_name in ['Cokgj', 'MultitaskGPR']:
            test_x_hf = [torch.empty(0, config.design.dimensionality), test_x.clone()]
            test_x_lf = [test_x.clone(), torch.empty(0, config.design.dimensionality)]

            observed_pred = surrogate.predict(test_x_hf)
            observed_pred_lf = surrogate.predict(test_x_lf)

            train_x = torch.tensor(train_data[-1].get_input_data().values)
            train_y = torch.tensor(scaler.inverse_transform(train_data[-1].get_output_data().values))

            train_x_lf = torch.tensor(train_data[0].get_input_data().values)
            train_y_lf = torch.tensor(scaler.inverse_transform(train_data[0].get_output_data().values))

        else:
            observed_pred = surrogate.predict(test_x)
            train_x = torch.tensor(train_data.get_input_data().values)
            train_y = torch.tensor(scaler.inverse_transform(train_data.get_output_data().values))
        
        exact_y = fun(test_x.clone())

        if config.design.dimensionality == 1 and config.visualize_gp:
            _, axs = plt.subplots(1, 1, num='gp' + config.regressor_name)

            surrogate.plot_gpr(
                test_x=test_x.clone().flatten(),
                scaler=scaler,
                exact_y=exact_y,
                observed_pred=observed_pred,
                color='b',
                train_x=train_x,
                train_y=train_y,
                savefig=True,
                axs=axs,
            )

            # if config.regressor_name in ['Cokgj', 'MultitaskGPR']:
            #     surrogate.plot_gpr(
            #         test_x=test_x.clone().flatten(),
            #         scaler=scaler,
            #         exact_y=None,
            #         observed_pred=observed_pred_lf,
            #         color='orange',
            #         train_x=train_x_lf,
            #         train_y=train_y_lf,
            #         savefig=True,
            #         axs=axs,
            #     )

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
    plt.title(metrics_df['2']['R^2_p'])
    plt.tight_layout()

cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()

plt.show()
