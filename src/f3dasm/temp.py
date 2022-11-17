import copy

import numpy as np

import f3dasm

seed = 2021
dim = 2
iterations = 1000
bounds = np.tile([-1.0, 1.0], (dim, 1))
design = f3dasm.make_nd_continuous_design(bounds=bounds, dimensionality=dim)

function = f3dasm.functions.Sphere(dimensionality=dim, scale_bounds=bounds, seed=seed, noise=0.1)

data = f3dasm.Data(design=design)
optimizer = f3dasm.optimization.Adam(data=data, seed=seed)
sampler = f3dasm.sampling.LatinHypercube(design=design, seed=seed)

samples = sampler.get_samples(30)
samples.add_output(output=function(samples))

optimizer.set_data(copy.copy(samples))
optimizer.iterate(iterations=iterations, function=function)
