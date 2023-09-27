#                                                                       Modules
# =============================================================================

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


class _SciPyOptimizer(Optimizer):
    type: str = 'scipy'

    def _callback(self, xk: np.ndarray, *args, **kwargs) -> None:
        self.data += ExperimentSample.from_numpy(xk)

    def update_step(self):
        """Update step function"""
        raise ValueError(
            'Scipy optimizers don\'t have an update steps. Multiple iterations \
                 are directly called througout scipy.minimize.')

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
            sample: ExperimentSample = data_generator.run(
                ExperimentSample.from_numpy(x))
            _, y = sample.to_numpy()
            return float(y)

        self.hyperparameters.maxiter = iterations

        minimize(
            fun=fun,
            method=self.method,
            # TODO: #89 Fix this with the newest gradient method!
            jac='3-point',
            x0=self.data.get_n_best_output(1).to_numpy()[0].ravel(),

            # x0=self.data.get_n_best_input_parameters_numpy(
            #     nosamples=1).ravel(),
            callback=self._callback,
            options=self.hyperparameters.__dict__,
            bounds=self.domain.get_bounds(),
            tol=0.0,
        )
