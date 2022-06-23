import numpy as np

from f3dasm.src.simulation import Function


class Levy(Function):
    """Levy function"""

    # def __init__(self):
    #     self.bounds = [-10, 10]

    def f(self, x: np.ndarray) -> np.ndarray:

        n_points, n_features = np.shape(x)
        y = np.empty((n_points, 1))

        for ii in range(n_points):
            z = 1 + (x[ii, :] - 1) / 4
            y[ii] = (
                np.sin(np.pi * z[0]) ** 2
                + sum((z[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * z[:-1] + 1) ** 2))
                + (z[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * z[-1]) ** 2)
            )
        return y
