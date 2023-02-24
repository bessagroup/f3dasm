import f3dasm
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

# torch.random.manual_seed(0)
# # Training data is 100 points in [0, 1] inclusive regularly spaced
# train_x = torch.linspace(0, 1, 100)
# # True function is sin(2*pi*x) with Gaussian noise
# train_y = train_x # + torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
# # train_y = (10.24 * (train_x - 0.5)) ** 2

class StandardScaler_torch:

    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def inverse_transform(self, values):
        return self.std * values + self.mean

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

###

dim = 1
seed = 123
noisy_data_bool = 1
numsamples = 50
fun_class = f3dasm.functions.Schwefel
opt_algo = torch.optim.Adam
training_iter = 50
kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()) \
     #+ gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
plot_mll = 1
plot_gpr = 1
train_surrogate = True
n_test = 500

###

fun = fun_class(
    dimensionality=dim,
    scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
    )

parameter_DesignSpace: f3dasm.DesignSpace = f3dasm.make_nd_continuous_design(
    bounds=np.tile([0.0, 1.0], (dim, 1)),
    dimensionality=dim,
)

sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=seed)

train_data: f3dasm.Data = sampler.get_samples(numsamples=numsamples)
# opt_retries = 50

output = fun(train_data) 
train_data.add_output(output=output)

train_x = torch.tensor(train_data.get_input_data().values)
train_y = torch.tensor(train_data.get_output_data().values.flatten())

scaler = StandardScaler()
scaler.fit(train_y.numpy()[:, None])
train_y_scaled = torch.tensor(scaler.transform(train_y.numpy()[:, None]).flatten())

train_y_scaled += noisy_data_bool * np.random.randn(*train_y_scaled.shape) * math.sqrt(0.04)

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()

if not noisy_data_bool:
    likelihood.noise_covar.register_constraint('raw_noise', gpytorch.constraints.Interval(1e-6, 1.1e-6))

# if noisy_data_bool:
#     likelihood = gpytorch.likelihoods.GaussianLikelihood()
# else:
#     cons = gpytorch.constraints.Interval(1e-6, 1.1e-6)
#     likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=cons)

model = ExactGPModel(train_x, train_y_scaled, likelihood)

# model.covar_module.kernels[0].base_kernel.register_constraint('raw_lengthscale', gpytorch.constraints.Interval(0, 0.3))
# model.covar_module.base_kernel.register_constraint('raw_lengthscale', gpytorch.constraints.Interval(0, 0.3))
# model.likelihood.noise_covar.register_constraint('raw_noise', gpytorch.constraints.Interval(0, 0.1))

def train(
    model, 
    likelihood, 
    opt_algo, 
    training_iter, 
    train_x, 
    train_y_scaled
    ):
    # Get into training mode to find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = opt_algo(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # losses = []
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calculate loss and backprop gradients
        loss = -mll(output, train_y_scaled.flatten())
        loss.backward()
        # losses.append(loss.item())
        if 1:#i % 10 == 0:
            print()
            print('Iter %d/%d' % (i + 1, training_iter), end=' - ')
            print('loss', "%.3f" % loss.item(), end=' - ')
            for k in range(len(list(model.parameters()))):
                print(
                    list(model.named_parameters())[k][0],
                    "%.3f" % list(model.constraints())[k].transform(list(model.parameters())[k]).flatten().item(),
                    end=' - '
                )
            # print(model.state_dict())
            # print('Iter %d/%d - Loss: %.3f outputscale1: %.3f lengthscale: %.3f noise: %.3f' % (
            #     i + 1, training_iter, loss.item(),
            #     model.covar_module.outputscale.item(),
            #     model.covar_module.base_kernel.lengthscale.item(),
            #     # model.covar_module.kernels[1].outputscale.item(),
            #     # model.covar_module.kernels[1].base_kernel.period_length.item(),
            #     model.likelihood.noise.item()
            # ))
        optimizer.step()
    # losses = np.array(losses)

    # print()
    # print(np.log10(losses[1:] / losses[:-1] + 1e-6))
    # print(np.median(np.log10(losses[1:] / losses[:-1])))

def gpr_plot(
    model, 
    likelihood, 
    train_x, 
    train_y,
    test_x,
    exact_y,
    observed_pred,
    ):

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1)#, figsize=(4, 3))

        # Get upper and lower confidence bounds
        lower, upper = [torch.tensor(scaler.inverse_transform(confbound.numpy()[:, None]).flatten()) for confbound in observed_pred.confidence_region()]
        # lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot ground truth as black dashed line
        ax.plot(test_x.numpy(), exact_y, 'k--')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), scaler.inverse_transform(observed_pred.mean.numpy()[:, None]).flatten(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        # ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Ground truth', 'Mean', 'Confidence'])
        plt.tight_layout()

def mll_plot(
    model, 
    likelihood,
    train_x, 
    train_y_scaled
    ):
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    opt_pars = [
        torch.tensor(model.likelihood.noise.item()),
        torch.tensor(model.covar_module.outputscale.item()),
        torch.tensor(model.covar_module.base_kernel.lengthscale.item())
        ]

    print()
    for i, (name, _) in enumerate(model.named_parameters()):
        print(name, opt_pars[i].item())

    model.train()
    likelihood.train()

    noise_level = torch.logspace(torch.log10(opt_pars[0]) - 0.1, torch.log10(opt_pars[0]) + 0.1, steps=30)
    amp_scale = torch.logspace(torch.log10(opt_pars[1]) - 0.5, torch.log10(opt_pars[1]) + 0.5, steps=30)
    length_scale = torch.logspace(torch.log10(opt_pars[2]) - 0.5, torch.log10(opt_pars[2]) + 0.5, steps=30)

    # length_scale_grid, noise_scale_grid = torch.meshgrid(length_scale, noise_level)
    length_scale_grid, amp_scale_grid = torch.meshgrid(length_scale, amp_scale)

    mll_plot_list = []
    # for scale, noise in zip(length_scale_grid.ravel(), noise_scale_grid.ravel()):
    for scale, amp in zip(length_scale_grid.ravel(), amp_scale_grid.ravel()):
        model.covar_module.base_kernel.lengthscale = scale
        # model.likelihood.noise = noise
        model.covar_module.outputscale = amp
        mll_plot_list.append(mll(model(train_x), train_y_scaled))

    mll_plot = torch.tensor(mll_plot_list).reshape(length_scale_grid.shape)
    mll_plot -= max(0, np.abs(np.amax(mll_plot.numpy()))) + 1e-3

    plot_min = [z for z in zip(length_scale_grid.ravel(), amp_scale_grid.ravel())][np.argmax(mll_plot)]
    
    from matplotlib.colors import LogNorm
    vmin, vmax = (-mll_plot).min(), (-mll_plot).max()
    level = np.logspace(np.log10(vmin), np.log10(vmax), num=50)

    plt.figure()
    plt.contourf(
        length_scale_grid.numpy(),
        # noise_scale_grid.numpy(),
        amp_scale_grid.numpy(),
        -mll_plot.numpy(),
        # levels=250,
        levels=level,
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    plt.colorbar()
    cs = plt.contour(
        length_scale_grid.numpy(),
        # noise_scale_grid.numpy(),
        amp_scale_grid.numpy(),
        -mll_plot.numpy(),
        # levels=250,
        levels=level,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        colors='black',
        linewidths=.3,
    )
    # plt.clabel(cs, cs.levels[::5], inline=1, fontsize=10)
    plt.plot(opt_pars[2], opt_pars[1], '.', color='white', label='Located optimum')
    plt.plot(plot_min[0], plot_min[1], 'r*', label='Optimum in range')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Length-scale")
    # plt.ylabel("Noise-level")
    plt.ylabel("Output-scale")
    plt.title("Log-marginal-likelihood")
    plt.legend()
    plt.tight_layout()

if train_surrogate:
    train(
        model=model, 
        likelihood=likelihood, 
        opt_algo=opt_algo, 
        training_iter=training_iter, 
        train_x=train_x, 
        train_y_scaled=train_y_scaled,
    )

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# # Test points are regularly spaced along [0,1]
# # Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=0)
    
    if dim == 1:
        test_x = torch.linspace(0, 1, n_test)
    else:
        test_x = torch.tensor(test_sampler.get_samples(numsamples=n_test).get_input_data().values)
    
    observed_pred = likelihood(model(test_x))
    exact_y = fun(test_x.numpy()[:, None])

def gp_metrics(observed_pred, exact_y):
    pred_mean = scaler.inverse_transform(observed_pred.mean.numpy()[:, None]).flatten()

    l1_norm = np.linalg.norm(exact_y - pred_mean, ord=1)
    l2_norm = np.linalg.norm(exact_y - pred_mean, ord=2)
    max_norm = np.linalg.norm(exact_y - pred_mean, ord=np.inf)

    # print()
    # print('NORMS', l1_norm, l2_norm, max_norm)

    MAE = l1_norm / len(exact_y)
    MSE = l2_norm ** 2 / len(exact_y)

    # print()
    # print('MEAN ERRORS', MAE, MSE, max_norm)

    sample_mean = np.mean(exact_y)
    sample_MAD = np.mean(np.abs(exact_y - sample_mean))
    sample_var = np.var(exact_y)
    sample_max_diff = np.linalg.norm(exact_y - sample_mean, ord=np.inf)

    # print()
    # print('SAMPLE STATS', sample_MAD, sample_var, sample_max_diff)

    FMU1 = MAE / sample_MAD
    FMU2 = MSE / sample_var
    FMUinf = max_norm / sample_max_diff

    Rsq1, Rsq2, Rsqinf = 1 - FMU1, 1 - FMU2, 1 - FMUinf

    # print()
    # print('COEFFICIENTS OF DETERMINATION', Rsq1, Rsq2, Rsqinf)

    metrics_array = np.array(
        [
            [l1_norm, l2_norm, max_norm],
            [MAE, MSE, max_norm],
            [sample_MAD, sample_var, sample_max_diff],
            [Rsq1, Rsq2, Rsqinf],
        ]
    )

    metrics_df = pd.DataFrame(
        data=metrics_array, 
        columns=['1', '2', 'inf'],
        index=['p-distance', 'mean p-error', 'sample p-deviation', 'R^2_p']
        )
    
    return metrics_df

metrics_df = gp_metrics(observed_pred=observed_pred, exact_y=exact_y)

print('\n')
print(metrics_df)

if plot_gpr and dim == 1:
    gpr_plot(
        model=model, 
        likelihood=likelihood,
        train_x=train_x, 
        train_y=torch.tensor(scaler.inverse_transform(train_y_scaled.numpy()[:, None]).flatten()),
        test_x=test_x,
        exact_y=exact_y,
        observed_pred=observed_pred,
        )

if plot_mll:
    mll_plot(
        model=model, 
        likelihood=likelihood,
        train_x=train_x, 
        train_y_scaled=train_y_scaled,
        )

###
if plot_gpr + plot_mll:
    plt.show()