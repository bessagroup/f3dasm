from typing import Any
import numpy as np
from f3dasm.src.designofexperiments import DoE
from f3dasm.src.simulation import Function
import pygmo as pg


class CMAES:
    def __init__(self, design: DoE, func: Function, seed: Any or int = None):
        self.design = design
        self.func = func
        self.seed = seed
        self.optimizer = pg.algorithm(
            pg.cmaes(gen=1, memory=True, seed=seed, force_bounds=True)
        )


class UDA_SGD:
    def __init__(self, step_size=1e-2, gradient_sparsity=1e-8):
        self.step_size = step_size
        self.gradient_sparsity = gradient_sparsity

    def evolve(self, pop):

        w = []

        # add numerical approximations of gradient as function evaluations
        for i in range(len(pop.problem.get_bounds()[0])):
            dx = np.zeros(len(pop.problem.get_bounds()[0]))
            dx[i] = self.gradient_sparsity
            w.append(pop.get_x()[pop.best_idx(), :] + dx)
            w.append(pop.get_x()[pop.best_idx(), :] - dx)

        g = pop.problem.gradient(pop.get_x()[pop.best_idx(), :])
        w.append(pop.get_x()[pop.best_idx(), :] - self.step_size * g)

        # check if decision vector is within bounds
        for ii in range(len(w)):
            for j in range(len(pop.problem.get_bounds())):
                if pop.problem.get_bounds()[0][j] > w[ii][j]:
                    w[ii][j] = pop.problem.get_bounds()[0][j]

                if pop.problem.get_bounds()[1][j] < w[ii][j]:
                    w[ii][j] = pop.problem.get_bounds()[1][j]

            pop.set_x(ii, w[ii])  # set updated step

        return pop

    def get_name(self):
        return "SGD"
