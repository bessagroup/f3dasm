import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import f3dasm
import gpytorch

df = pd.read_csv('notebooks/resources/sweep_MeshSize=1.55_20221227_1506.csv')
df_clean = df.dropna()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df_clean['WThk'], df_clean['FL'], df_clean['MaxEPS'])

###

dimensionality = 2

design: f3dasm.DesignSpace = f3dasm.make_nd_continuous_design(
    bounds=np.tile([0.0, 1.0], (dimensionality, 1)),
    dimensionality=dimensionality,
)

train_data = f3dasm.Data(design=design)
train_data.add_numpy_arrays(input=df_clean[['WThk', 'FL']].values, output=df_clean['MaxEPS'].values[:, None])

noise_fix = True

kernel = gpytorch.kernels.ScaleKernel(base_kernel=gpytorch.kernels.RBFKernel())
param = f3dasm.regression.gpr.Sogpr_Parameters(kernel=kernel)

regressor = f3dasm.regression.gpr.Sogpr(
    train_data=train_data, 
    design=train_data.design,
    parameter=param,
    noise_fix=noise_fix,
)

surrogate = regressor.train()

x0_plot = np.linspace(0.1, 0.49, 50)
x1_plot = np.linspace(0.5, 1.2, 50)
x_plot = np.meshgrid(*[x0_plot, x1_plot])
x_plot_array = np.vstack((x_plot[0].flatten(), x_plot[1].flatten())).T

x_plot_data = f3dasm.Data(design=design)
x_plot_data.add_numpy_arrays(input=x_plot_array, output=x_plot_array[:, 0][:, None])

mean, var = surrogate.predict(test_input_data=x_plot_data)

ax.plot_surface(x_plot[0], x_plot[1], mean.reshape((50, 50)), cmap='viridis', alpha=0.5)

plt.show()