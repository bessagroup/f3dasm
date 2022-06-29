from typing import Any
import numpy as np
from f3dasm.src.designofexperiments import DoE
from f3dasm.src.simulation import Function
import pygmo as pg


class PygmoProblem:
    """Convert a testproblem from problemset to a pygmo object"""

    def __init__(self, design: DoE, func: Function, seed: Any or int = None):
        self.design = design
        self.func = func
        self.seed = seed

        if seed:
            pg.set_global_rng_seed(seed)

    def fitness(self, x: np.ndarray) -> np.ndarray:
        return self.func.eval(x).ravel()  # pygmo doc: should output 1D numpy array

    def batch_fitness(self, x: np.ndarray) -> np.ndarray:
        return self.func.eval(x).ravel()

    def get_bounds(self) -> tuple:
        return (
            [
                parameter.lower_bound
                for parameter in self.design.get_continuous_parameters()
            ],
            [
                parameter.upper_bound
                for parameter in self.design.get_continuous_parameters()
            ],
        )

    def gradient(self, x: np.ndarray):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)


# TO DO: make dataclass ?
class UDA_Adam:  # step_size=1e-2
    def __init__(
        self, step_size=1e-2, gradient_sparsity=1e-8, beta_1=0.9, beta_2=0.999
    ):
        self.m = 0
        self.v = 0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.step_size = step_size
        self.gradient_sparsity = gradient_sparsity

    def evolve(self, pop, epsilon=1e-8):

        w = []

        # add numerical approximations of gradient as function evaluations
        for i in range(len(pop.problem.get_bounds()[0])):
            dx = np.zeros(len(pop.problem.get_bounds()[0]))
            dx[i] = self.gradient_sparsity
            w.append(pop.get_x()[pop.best_idx(), :] + dx)
            w.append(pop.get_x()[pop.best_idx(), :] - dx)

        g = pop.problem.gradient(pop.get_x()[pop.best_idx(), :])
        t = (
            int(round(pop.problem.get_fevals() / len(pop) - 1)) + 1
        )  # minus 1 to counter f_evals counter when pop is created
        m = self.beta_1 * self.m + (1 - self.beta_1) * g
        v = self.beta_2 * self.v + (1 - self.beta_2) * np.power(g, 2)
        m_hat = m / (1 - np.power(self.beta_1, t))
        v_hat = v / (1 - np.power(self.beta_2, t))

        w.append(
            pop.get_x()[pop.best_idx(), :]
            - self.step_size * m_hat / (np.sqrt(v_hat) + epsilon)
        )

        # check if decision vector is within bounds
        for ii in range(len(w)):
            for j in range(len(pop.problem.get_bounds())):
                if pop.problem.get_bounds()[0][j] > w[ii][j]:
                    w[ii][j] = pop.problem.get_bounds()[0][j]

                if pop.problem.get_bounds()[1][j] < w[ii][j]:
                    w[ii][j] = pop.problem.get_bounds()[1][j]

            pop.set_x(ii, w[ii])  # set updated step

        self.m = m
        self.v = v

        return pop

    def get_name(self):
        return "Adam"
