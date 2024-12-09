#                                                                       Modules
# =============================================================================

# Standard
import warnings
from typing import Protocol

# Third-party core
import autograd.numpy as np
from scipy.optimize import minimize

# Locals
from ...datageneration.datagenerator import DataGenerator
from ...experimentdata.experimentsample import ExperimentSample
from ..optimizer import Optimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

warnings.filterwarnings(
    "ignore", message="^OptimizeWarning: Unknown solver options.*")


class ExperimentData(Protocol):
    ...


class ScipyOptimizer(Optimizer):
    require_gradients: bool = False
    type: str = 'scipy'

    def __init__(self, algorithm_cls, **hyperparameters):
        self.algorithm_cls = algorithm_cls
        self.hyperparameters = hyperparameters

    def init(self, data: ExperimentData, data_generator: DataGenerator):
        self.data = data
        self.algorithm = self.algorithm_cls
        self.data_generator = data_generator

    def _callback(self, xk: np.ndarray, *args, **kwargs) -> None:
        self.data.add_experiments(
            ExperimentSample.from_numpy(xk, domain=self.data.domain))

    def update_step(self):
        """Update step function"""
        raise ValueError(
            'Scipy optimizers don\'t have an update steps. \
                 Multiple iterations are directly called \
                    througout scipy.minimize.')

    def run_algorithm(self, iterations: int):
        """Run the algorithm for a number of iterations

        Parameters
        ----------
        iterations
            number of iterations
        """

        def fun(x):
            sample: ExperimentSample = self.data_generator._run(
                x, domain=self.data.domain)
            _, y = sample.to_numpy()
            return float(y)

        if not hasattr(self.data_generator, 'dfdx'):
            self.data_generator.dfdx = None

        self.hyperparameters['maxiter'] = iterations

        minimize(
            fun=fun,
            method=self.algorithm,
            jac=self.data_generator.dfdx,
            x0=self.data.get_n_best_output(1).to_numpy()[0].ravel(),
            callback=self._callback,
            options=self.hyperparameters,
            bounds=self.data.domain.get_bounds(),
            tol=0.0,
        )
