import matplotlib.pyplot as plt
import numpy as np

import f3dasm

dim = 1
fun = f3dasm.functions.Sphere(dimensionality=dim)
x = np.array([[0]])
x = fun._from_input_to_scaled(x)