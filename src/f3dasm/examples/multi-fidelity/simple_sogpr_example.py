import matplotlib.pyplot as plt
import numpy as np

import f3dasm

dim = 1

fun_class = f3dasm.functions.AlpineN2

print(fun_class.is_dim_compatible(dim))

fun = fun_class(
    dimensionality=dim,
    scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
    )

parameter_DesignSpace: f3dasm.DesignSpace = f3dasm.make_nd_continuous_design(
    # bounds=fun.input_domain.astype(float),
    bounds=np.tile([0.0, 1.0], (dim, 1)),
    dimensionality=dim,
)

sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace)

train_data: f3dasm.Data = sampler.get_samples(numsamples=8)

train_data.add_output(output=fun(train_data))

print(train_data)

regressor = f3dasm.regression.gpr.Sogpr(
    train_data=train_data, 
    design=train_data.design,
)

surrogate = regressor.train()

test_input_data: f3dasm.Data = sampler.get_samples(numsamples=500)
mean, var = surrogate.predict(test_input_data=test_input_data)

# print(mean)

# plt.scatter(test_input_data.data['input'], mean)
# plt.show()