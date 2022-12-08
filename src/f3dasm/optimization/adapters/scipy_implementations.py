#                                                                       Modules
# =============================================================================

# Standard
from typing import Tuple

# Third-party
import autograd.numpy as np
from scipy.optimize import minimize

# Locals
from ...base.function import Function
from ...base.optimization import Optimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class SciPyOptimizer(Optimizer):
    def _callback(self, xk: np.ndarray, *args, **kwargs) -> None:
        self.x_new.append(xk.tolist())

    def update_step(self):
        """Update step function"""
        raise ValueError(
            'Scipy optimizers don\'t have an update steps. Multiple iterations \
                 are directly called througout scipy.minimize.')

    def run_algorithm(self, iterations: int, function: Function):
        """Run the algorithm for a number of iterations

        :param iterations: number of iterations
        :param function: function to be evaluated
        """
        pass

    def iterate(self, iterations: int, function: Function):
        """Iterating on a funtion

        :param iterations: number of iterations
        :param function: function to be evaluated
        """
        self.x_new = []

        self.parameter.maxiter = iterations

        self.run_algorithm(iterations, function)

        self.x_new = np.array(self.x_new)

        # If x_new is empty, repeat best x0 to fill up total iteration
        if len(self.x_new) == 0:
            repeated_last_element = np.tile(
                self.data.get_n_best_input_parameters_numpy(
                    nosamples=1).ravel(),
                (iterations - len(self.x_new), 1),
            )
            self.x_new = repeated_last_element

        # Repeat last iteration to fill up total iteration
        if len(self.x_new) < iterations:
            repeated_last_element = np.tile(
                self.x_new[-1], (iterations - len(self.x_new), 1))
            self.x_new = np.r_[self.x_new, repeated_last_element]

        self.add_iteration_to_data(x=self.x_new, y=function(self.x_new))


class SciPyMinimizeOptimizer(SciPyOptimizer):
    def run_algorithm(self, iterations: int, function: Function):
        """Run the algorithm for a number of iterations

        :param iterations: number of iterations
        :param function: function to be evaluated
        """
        minimize(
            fun=lambda x: function(x).item(),
            method=self.method,
            jac=lambda x: function.dfdx(x).ravel(),
            x0=self.data.get_n_best_input_parameters_numpy(
                nosamples=1).ravel(),
            callback=self._callback,
            options=self.parameter.__dict__,
            bounds=function.scale_bounds,
            tol=0.0,
        )
