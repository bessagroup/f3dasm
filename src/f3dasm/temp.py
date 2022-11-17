import numpy as np

import f3dasm

dim = 3
iterations = 50
realizations = 3
bounds = np.tile([-1.0, 1.0], (dim, 1))
hyperparameters = {}
# hyperparameters={'learning_rate': 1e-2}

design = f3dasm.make_nd_continuous_design(bounds=bounds, dimensionality=dim)

function = f3dasm.functions.Levy(dimensionality=dim, scale_bounds=bounds, seed=42)

data = f3dasm.Data(design=design)
optimizer = f3dasm.optimization.CG(data=data, hyperparameters=hyperparameters)
sampler = f3dasm.sampling.LatinHypercube(design=design)
