import f3dasm
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
from typing import Any

from sklearn.preprocessing import StandardScaler

from f3dasm.design import ExperimentData

from f3dasm.functions.fidelity_augmentors import NoiseInterpolator, Scale, NoisyBaseFun, BiasedBaseFun

###

dim = 1
seed = 123
noisy_data_bool = 1
n_test = 50
train_data_supplied = 0

mean_module_list = torch.nn.ModuleList([
    gpytorch.means.ZeroMean(),
    gpytorch.means.ZeroMean()
])

covar_module_list = torch.nn.ModuleList([
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
])

base_fun = f3dasm.functions.AlpineN2(
    dimensionality=dim,
    scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
    offset=False,
)

def scale_factor(input_row: np.ndarray) -> np.ndarray:
    try:
        res = input_row.get_input_data().values + .5
    except:
        res = input_row + .5
    return res

def scale(input_row: np.ndarray, output_row: np.ndarray) -> np.ndarray:
    output_row *= scale_factor(input_row=input_row)
    return output_row

def reverse_scale(input_row: np.ndarray, output_row: np.ndarray) -> np.ndarray:
    output_row /= scale_factor(input_row=input_row)
    return output_row

noise_augmentor = NoiseInterpolator(noise_var=1., aug_coeff=0.75)
scale_augmentor = Scale(scale=scale, reverse_scale=reverse_scale)

aug_fun = NoisyBaseFun(
    seed=seed,
    dimensionality=dim,
    base_fun=base_fun,
    noise_augmentor=noise_augmentor,
)

# aug_fun = BiasedBaseFun(
#     seed=seed,
#     dimensionality=dim,
#     base_fun=base_fun,
#     scale_augmentor=scale_augmentor,
# )

###

fids = [0.5, 1.0]
samp_nos = [20, 5]

funs = []
train_x_list = []
train_y_scaled_list = []
mf_design_space = []
mf_sampler = []
mf_train_data = []

for fid_no, (fid, samp_no) in enumerate(zip(fids, samp_nos)):

    if fid_no == 0:
        fun = aug_fun
    else:
        fun = base_fun
    
    parameter_DesignSpace = f3dasm.make_nd_continuous_design(
        bounds=np.tile([0.0, 1.0], (dim, 1)),
        dimensionality=dim,
    )

    sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=seed)

    if train_data_supplied:
        train_data = ExperimentData(design=parameter_DesignSpace)

        if fid_no == 0:
            input_array = np.linspace(0, 1, 11)[:, None]
        else:
            input_array = np.array([0., 0.4, 0.6, 1.])[:, None]

        train_data.add_numpy_arrays(
            input_rows=input_array, 
            output_rows=np.full_like(input_array, np.nan)
            )
    else:
        train_data = sampler.get_samples(numsamples=samp_no)

    train_data.add_output(fun(train_data))

    train_x = torch.tensor(train_data.get_input_data().values)
    train_y = torch.tensor(train_data.get_output_data().values.flatten())

    train_x_list.append(train_x)

    # Scaling the output
    if fid_no == 0:
        scaler = StandardScaler()
        scaler.fit(train_y.numpy()[:, None])
    train_y_scaled = torch.tensor(scaler.transform(train_y.numpy()[:, None]).flatten())

    # if fid_no == 0:
    #     train_y_scaled += noisy_data_bool * np.random.randn(*train_y_scaled.shape) * math.sqrt(0.04)

    train_y_scaled_list.append(train_y_scaled)

    train_data.add_output(output=train_y_scaled)

    funs.append(fun)
    mf_design_space.append(parameter_DesignSpace)
    mf_sampler.append(sampler)
    mf_train_data.append(train_data)

###

param = f3dasm.machinelearning.gpr.Cokgj_Parameters(
    likelihood=gpytorch.likelihoods.GaussianLikelihood(),
    kernel=covar_module_list,
    mean=mean_module_list,
    noise_fix=1 - noisy_data_bool,
    opt_algo=torch.optim.Adam,
    opt_algo_kwargs=dict(lr=0.1),
    verbose_training=1,
    training_iter=150,
    )

regressor = f3dasm.machinelearning.gpr.Cokgj(
    train_data=mf_train_data, 
    design=train_data.design,
    parameter=param,
)

surrogate = regressor.train()

# print()
# print(dict(surrogate.model.named_parameters()))

###

# Get into evaluation (predictive posterior) mode
surrogate.model.eval()
surrogate.model.likelihood.eval()

# # Test points are regularly spaced along [0,1]
# # Make predictions by feeding model through likelihood

test_sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=0)

if dim == 1:
    test_x = torch.linspace(0, 1, n_test)[:, None]
else:
    test_x = torch.tensor(test_sampler.get_samples(numsamples=n_test).get_input_data().values)


test_x_hf = [torch.empty(0, dim), test_x]
test_x_lf = [test_x, torch.empty(0, dim)]

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred_hf = surrogate.predict(test_x_hf)
    observed_pred_lf = surrogate.predict(test_x_lf)

exact_y_hf = funs[1](test_x.numpy())#[:, None])
exact_y_lf = funs[0](test_x.numpy())

###

# Show VFUCB acquisition if needed. First define the acquisition function:
acquisition_class = f3dasm.machinelearning.acquisition_functions.VFUpperConfidenceBound
acquisition_hyperparameters = dict(beta=0.4, maximize=False)
mean_low = observed_pred_lf.mean
mean_high = observed_pred_hf.mean

acquisition = acquisition_class(
    model=surrogate.model,
    mean=[mean_low, mean_high],
    cr=20,
    **acquisition_hyperparameters
)

if dim == 1:
    f, axs = plt.subplots(2, 1, num='gp and acq', figsize=(6, 8), sharex='all')

    surrogate.plot_gpr(
        test_x=test_x.flatten(), 
        scaler=scaler, 
        exact_y=exact_y_hf, 
        observed_pred=observed_pred_hf,
        train_x=train_x_list[1],
        train_y=torch.tensor(scaler.inverse_transform(train_y_scaled_list[1].numpy()[:, None])),
        color='b',
        acquisition=acquisition,
        fid=1,
        axs=axs,
        )
    
    surrogate.plot_gpr(
        test_x=test_x.flatten(), 
        scaler=scaler, 
        # exact_y=exact_y_lf, 
        observed_pred=observed_pred_lf,
        train_x=train_x_list[0],
        train_y=torch.tensor(scaler.inverse_transform(train_y_scaled_list[0].numpy()[:, None])),
        color='orange',
        acquisition=acquisition,
        fid=0,
        axs=axs,
        )
    
    axs[0].set_ylabel('y')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('acquisition')

    plt.tight_layout()


# surrogate.plot_mll(
#     train_x=train_x,
#     train_y_scaled=train_y_scaled,
# )

metrics_df = surrogate.gp_metrics(
    scaler=scaler,
    observed_pred=observed_pred_hf,
    exact_y=exact_y_hf.flatten(),
)

print()
print(metrics_df)

plt.show()