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

from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)


def convert_config_to_input(config: Config) -> List[dict]:

    seed = np.random.randint(low=0, high=1e5)
    # seed = 123

    with open('seed.txt', "w") as f:
        f.write(str(seed))

    bounds = np.tile([0.0, 1.0], (config.design.dimensionality, 1))
    design = f3dasm.make_nd_continuous_design(bounds=bounds, dimensionality=config.design.dimensionality)

    regressor_class = f3dasm.find_class(f3dasm.machinelearning.gpr, config.regressor.regressor_name)

    sampler_class: f3dasm.Sampler = f3dasm.find_class(f3dasm.sampling, config.sampler.sampler_name)
    sampler = sampler_class(design=design, seed=seed)

    kernel = gpytorch.kernels.ScaleKernel(f3dasm.find_class(gpytorch.kernels, config.regressor.kernel_name)())

    mean = f3dasm.find_class(gpytorch.means, config.regressor.mean_name)()

    param = f3dasm.machinelearning.gpr.Sogpr_Parameters(
        kernel=kernel,
        mean=mean,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        noise_fix=1 - config.function.noisy,
        opt_algo=torch.optim.Adam,
        opt_algo_kwargs=dict(lr=0.1),
        verbose_training=False,
        training_iter=50,
        )

    regressors = []
    funs = []
    scalers = []
    for name in config.function.function_names:
        torch.manual_seed(seed=seed)

        fun_class: f3dasm.Function = f3dasm.find_class(f3dasm.functions, name) 
        fun = fun_class(dimensionality=config.design.dimensionality, scale_bounds=bounds, seed=seed)

        ## Generate training samples with the sampler
        train_data: f3dasm.ExperimentData = sampler.get_samples(numsamples=config.sampler.number_of_samples)
        fun(train_data)

        ## Extracting output data for scaling
        train_y = torch.tensor(train_data.get_output_data().values)

        ## Scaling the training data output
        scaler = StandardScaler()
        scaler.fit(train_y.numpy())
        train_y_scaled = torch.tensor(scaler.transform(train_y.numpy()))

        ## Add Gaussian noise to the data if specified
        np.random.seed(seed=seed)
        train_y_scaled += config.function.noisy * np.random.randn(*train_y_scaled.shape) * np.sqrt(0.04)

        ## Scaled data added to the training data
        train_data.add_output(output=train_y_scaled)
        
        ## Define regressor
        regressor = regressor_class(
            train_data=train_data,
            parameter=param,
        )

        regressors.append(regressor)
        funs.append(fun)
        scalers.append(scaler)

    options_list = [
        {
        "regressor": regressor,
        "function": fun,
        "design": design,
        "scaler": scaler,
        } for regressor, fun, scaler in zip(regressors, funs, scalers)
        ]

    return options_list

@hydra.main(config_path=".", config_name="config")
def main(cfg: Config):
    options_list = convert_config_to_input(config=cfg)

    for options in options_list:
        regressor = options['regressor']
        torch.manual_seed(123)
        surrogate = regressor.train()

        n_test = 500

        ## Get into evaluation (predictive posterior) mode
        surrogate.model.eval()
        surrogate.model.likelihood.eval()

        ## Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_sampler = f3dasm.sampling.SobolSequence(design=options['design'], seed=0)

            test_x = torch.tensor(test_sampler.get_samples(numsamples=n_test).get_input_data().values)
            # test_x = torch.linspace(0, 1, n_test)
            
            observed_pred = surrogate.predict(test_x)
            exact_y = options['function'](test_x.numpy()[:, None])
        
        ## Calculate and store metrics
        metrics_df = surrogate.gp_metrics(
            scaler=options['scaler'],
            observed_pred=observed_pred,
            exact_y=exact_y.flatten(),
        )

        metrics_df.to_csv(options['function'].name + '.csv')

cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
