"""
Experiment to neurally reparameterize a chosen benchmark function.
"""
import copy

import autograd.numpy as np
import tensorflow as tf

# imports
import f3dasm

# Create a model
seed = 0
shape = (1, 2)
model = f3dasm.machinelearning.neural_repara_models.GenericModel(seed, shape)

# Create a benchmark function to test on
dim = shape[0] * shape[1]
domain = np.tile([-1e5, 1e5], (dim, 1))
design = f3dasm.make_nd_continuous_domain(bounds=domain, dimensionality=dim)
ackley_fn = f3dasm.functions.pybenchfunction.Ackley(dimensionality=dim,
                                                    seed=seed, scale_bounds=domain)
# Choose an optimizer
sampler = f3dasm.sampling.LatinHypercube(domain=design, seed=seed)
N = 1  # Number of samples
data = sampler.get_samples(numsamples=N)

# set weights to data object
data.reset_data()
start_weights = model.get_model_weights()
data.add_numpy_arrays(start_weights.T, np.array([[np.nan]]))

opt = f3dasm.optimization.Adam(data=copy.copy(data), seed=seed)
# Run the optimization
max_iterations = 100
for i in range(max_iterations):
    with tf.GradientTape() as t:
        t.watch(model.trainable_variables)
        logits = model(None)
        loss = ackley_fn.evaluate(logits)
    grads = t.gradient(loss, model.trainable_variables)
    # Apply optimzier step
    opt.update(model, grads)
    # or
    opt.algorithm.apply_gradients((grads, model.trainable_variables))


# %%
