import f3dasm
import numpy as np
function = f3dasm.functions.Branin
scale_bounds_list = [-1.0, 1.0]

dim = 6
if not function.is_dim_compatible(dim):
    dim = 4
    if not function.is_dim_compatible(dim):
        dim = 3
        if not function.is_dim_compatible(dim):
            dim = 2

seed = np.random.randint(low=0, high=1e5)
scale_bounds = np.tile(scale_bounds_list, (dim, 1))
func = function(noise=False, seed=seed, scale_bounds=scale_bounds, dimensionality=dim)



if func.get_global_minimum(func.dimensionality)[0] is not None:
    assert func.check_if_within_bounds(func.get_global_minimum(func.dimensionality)[0])