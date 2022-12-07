import f3dasm
import numpy as np
from numdifftools import Gradient

dimensionality = 2
design = f3dasm.make_nd_continuous_design(bounds=np.tile(
    [-1., 1.], (dimensionality, 1)), dimensionality=dimensionality)


function = f3dasm.functions.Ackley(
    dimensionality=dimensionality, scale_bounds=design.get_bounds())


function.dfdx(np.array([0.1, 0.1]))
