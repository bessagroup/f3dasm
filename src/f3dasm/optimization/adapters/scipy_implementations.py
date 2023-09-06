#                                                                       Modules
# =============================================================================

# Third-party core
import autograd.numpy as np
from scipy.optimize import minimize

# Locals
from ...datageneration.functions import Function
from ...design.design import Design
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
    def _callback(self, xk: np.ndarray, *args, **kwargs) -> None:
        self.data.add_design(Design.from_numpy(xk))

    def update_step(self):
        """Update step function"""
        raise ValueError(
            'Scipy optimizers don\'t have an update steps. Multiple iterations \
                 are directly called througout scipy.minimize.')

    def run_algorithm(self, iterations: int, function: Function):
        """Run the algorithm for a number of iterations

        Parameters
        ----------
        iterations
            number of iterations
        function
            function to be evaluated
        """
        pass

    def iterate(self, iterations: int, function: Function):
        """Iterating on a function

        Parameters
        ----------
        iterations
            number of iterations
        function
            function to be evaluated
        """
        n_data_before_iterate = len(self.data)

        self.parameter.maxiter = iterations

        self.run_algorithm(iterations, function)

        # If x_new is empty, repeat best x0 to fill up total iteration
        if len(self.data) == n_data_before_iterate:
            repeated_last_element = self.data.get_n_best_input_parameters_numpy(
                nosamples=1).ravel()

            for repetition in range(iterations):
                self._callback(repeated_last_element)

        # Repeat last iteration to fill up total iteration
        if len(self.data) < n_data_before_iterate + iterations:
            last_design = self.data.get_design(len(self.data)-1)

            for repetition in range(iterations - (len(self.data) - n_data_before_iterate)):
                self.data.add_design(last_design)

        # Evaluate the function on the extra iterations
        self.data.run(function.run)


class _SciPyMinimizeOptimizer(_SciPyOptimizer):
    def run_algorithm(self, iterations: int, function: Function):
        """Run the algorithm for a number of iterations

        Parameters
        ----------
        iterations
            number of iterations
        function
            function to be evaluated
        """

        # check if fun has the original_function attribute
        if hasattr(function, 'original_function'):
            # Convert the input of the original function to an ndarray using a lambda function
            def fun(x):
                # convert the np.ndarray input to a dict with key x0, x1, x2, etc.
                input_names = self.data.domain.get_continuous_names()
                x = {input_names[i]: x_i for i, x_i in enumerate(x)}
                return function.original_function(x)

        else:
            def fun(x):
                return function(x).item()
        minimize(
            fun=fun,
            method=self.method,
            # TODO: #89 Fix this with the newest gradient method!
            jac=lambda x: np.float64(function.dfdx_legacy(x).ravel()),
            x0=self.data.get_n_best_input_parameters_numpy(
                nosamples=1).ravel(),
            callback=self._callback,
            options=self.parameter.__dict__,
            bounds=self.data.domain.get_bounds(),
            tol=0.0,
        )
