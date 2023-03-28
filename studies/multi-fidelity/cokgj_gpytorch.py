import f3dasm
import torch
import gpytorch

train_x = [
    torch.tensor([
        [0.1],
        [0.6],
        [0.8],
        [0.75],
        ]),
    torch.tensor([
        [0.5],
        [0.3],
        ])
    ]

train_y = [
    torch.tensor([
        [0.5],
        [0.8],
        [0.0],
        [0.05],
        ]),
    torch.tensor([
        [1.2],
        [1.1],
        ])
    ]

likelihood = gpytorch.likelihoods.GaussianLikelihood()

mean_module_list = [
    gpytorch.means.ZeroMean(),
    gpytorch.means.ZeroMean()
]

covar_module_list = [
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()),
]

cokgj_model = f3dasm.regression.CokgjModel(
    train_x=train_x, 
    train_y=train_y,
    likelihood=likelihood,
    mean_module_list=mean_module_list,
    covar_module_list=covar_module_list,
    )

dist = cokgj_model.forward(x=train_x)
print(dist)