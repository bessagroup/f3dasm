from typing import Callable
import numpy as np

from f3dasm.src.simulation import Function


def create_test_data(f: Callable):
    x = np.array([0.2, 0.3, 0.4, 0.6])  # 1D array with 4 dimensions
    x1 = np.array([[0.1], [0.2], [0.3]])  # 2D array with 1 dimension
    x2 = np.array([[0.0, 0.0], [0.5, 0.3], [1.0, 0.8]])  # 2D array with 2 dimensions
    x3 = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.3, 0.2], [1.0, 0.8, 0.5]]
    )  # 2D array with 3 dimensions
    out = [
        ("x", repr(f.eval(x))),
        ("x1", repr(f.eval(x1))),
        ("x2", repr(f.eval(x2))),
        ("x3", repr(f.eval(x3))),
    ]
    print(out)


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


class Ackley(Function):
    """Ackley function"""

    def f(
        self, x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi
    ) -> np.ndarray:

        n_points, n_features = np.shape(x)
        y = np.empty((n_points, 1))

        for ii in range(n_points):

            s1 = sum(x[ii, :] ** 2)
            s2 = sum(np.cos(c * x[ii, :]))
            y[ii] = (
                -a * np.exp(-b * np.sqrt(s1 / n_features))
                - np.exp(s2 / n_features)
                + a
                + np.exp(1)
            )
        return y


class Rosenbrock(Function):
    """Rosenbrock function"""

    def f(self, x: np.ndarray) -> np.ndarray:

        n_points, n_features = np.shape(x)
        y = np.empty((n_points, 1))
        for ii in range(n_points):
            x0 = x[ii, :-1]
            x1 = x[ii, 1:]
            y[ii] = sum((1 - x0) ** 2) + 100 * sum((x1 - x0**2) ** 2)
        return y


class Schwefel(Function):
    """Schwefel function"""

    def f(self, x: np.ndarray) -> np.ndarray:

        n_points, n_features = np.shape(x)
        y = np.empty((n_points, 1))
        for ii in range(n_points):
            y[ii] = 418.9829 * n_features - sum(
                x[ii, :] * np.sin(np.sqrt(abs(x[ii, :])))
            )
        return y


class Rastrigin(Function):
    """Rastrigin function"""

    def f(self, x: np.ndarray) -> np.ndarray:
        n_points, n_features = np.shape(x)
        y = np.empty((n_points, 1))

        for ii in range(n_points):
            y[ii] = 10 * n_features + sum(
                x[ii, :] ** 2 - 10 * np.cos(2 * np.pi * x[ii, :])
            )
        return y
