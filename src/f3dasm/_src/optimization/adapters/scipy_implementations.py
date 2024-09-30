#                                                                       Modules
# =============================================================================

# Standard
import warnings

# Third-party core
import autograd.numpy as np
from scipy.optimize import minimize

# Locals
from ...datageneration.datagenerator import DataGenerator
from ...design.domain import Domain
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


class _SciPyOptimizer(Optimizer):
    type: str = 'scipy'

    def __init__(self, domain: Domain, method: str, **hyperparameters):
        self.domain = domain
        self.method = method
        self.options = {**hyperparameters}

    def _callback(self, xk: np.ndarray, *args, **kwargs) -> None:
        self.data.add_experiments(
            ExperimentSample.from_numpy(xk, domain=self.domain))

    def update_step(self):
        """Update step function"""
        raise ValueError(
            'Scipy optimizers don\'t have an update steps. \
                 Multiple iterations are directly called \
                    througout scipy.minimize.')

    def run_algorithm(self, iterations: int, data_generator: DataGenerator):
        """Run the algorithm for a number of iterations

        Parameters
        ----------
        iterations
            number of iterations
        function
            function to be evaluated
        """

        def fun(x):
            sample: ExperimentSample = data_generator._run(
                x, domain=self.domain)
            _, y = sample.to_numpy()
            return float(y)

        self.options['maxiter'] = iterations

        if not hasattr(data_generator, 'dfdx'):
            data_generator.dfdx = None

        minimize(
            fun=fun,
            method=self.method,
            jac=data_generator.dfdx,
            x0=self.data.get_n_best_output(1).to_numpy()[0].ravel(),
            callback=self._callback,
            options=self.options,
            bounds=self.domain.get_bounds(),
            tol=0.0,
        )
