#                                                                       Modules
# =============================================================================

# Third-party core
import autograd.numpy as np
from scipy.optimize import minimize

# Locals
from ...datageneration.functions import Function
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
        self.x_new.append(xk.tolist())

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

        # check if fun has the original_function attribute
        if hasattr(function, 'original_function'):
            # Convert the input of the original function to an ndarray using a lambda function
            def fun(x):
                # convert the np.ndarray input to a dict with key x0, x1, x2, etc.
                input_names = self.data.design.get_continuous_input_names()
                x = {input_names[i]: x_i for i, x_i in enumerate(x)}
                return function.original_function(x)

        else:
            def fun(x):
                return function(x).item()

        _y = []
        for _x in self.x_new:
            _y.append(fun(_x))

        _y = np.array(_y).reshape(-1, 1)

        # self.add_iteration_to_data(x=self.x_new, y=function(self.x_new))
        self.add_iteration_to_data(x=self.x_new, y=_y)


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
                input_names = self.data.design.get_continuous_input_names()
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
            bounds=self.data.design.get_bounds(),
            tol=0.0,
        )
