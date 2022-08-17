import numpy as np

from ..base.optimization import Optimizer
from ..base.function import Function


class RandomSearch(Optimizer):
    """Naive random search"""

    def update_step(self, function: Function) -> None:

        x_new = np.atleast_2d(
            [
                np.random.uniform(low=function.scale_bounds[d, 0], high=function.scale_bounds[d, 1])
                for d in range(function.dimensionality)
            ]
        )

        self.data.add_numpy_arrays(input=x_new, output=function.__call__(x_new))


class SGD(Optimizer):
    """Gradient-based Stochastig Gradient Descent (SGD) optimizer"""

    def init_parameters(self):

        # Default hyperparameters
        self.defaults = {"learning_rate": 1e-2, "force_bounds": True}

    def update_step(self, function: Function) -> None:

        x = self.data.get_input_data().iloc[-self.hyperparameters["population"] :].to_numpy()

        g = function.dfdx(x)

        x_new = x - self.hyperparameters["learning_rate"] * g

        # Force bounds
        if self.hyperparameters["force_bounds"]:
            x_new = self._force_bounds(x_new, function.scale_bounds)

        self.data.add_numpy_arrays(input=x_new, output=function(x_new))


class Momentum(Optimizer):
    """Gradient-based Momentum optimizer"""

    def init_parameters(self):

        # Default hyperparameters
        self.defaults = {
            "learning_rate": 1e-2,
            "beta": 0.9,
            "force_bounds": True,
        }

        # Dynamic parameters
        self.m = 0

    def update_step(self, function: Function) -> None:

        x = self.data.get_input_data().iloc[-self.hyperparameters["population"] :].to_numpy()

        g = function.dfdx(x)

        m = self.hyperparameters["beta"] * self.m + (1 - self.hyperparameters["beta"]) * g
        x_new = x - self.hyperparameters["learning_rate"] * m

        # Force bounds
        if self.hyperparameters["force_bounds"]:
            x_new = self._force_bounds(x_new, function.scale_bounds)

        # Update dynamic parameters
        self.m = m

        self.data.add_numpy_arrays(input=x_new, output=function(x_new))


class Adam(Optimizer):
    """Gradient-based Adam optimizer"""

    def init_parameters(self):

        # Default hyperparameters
        self.defaults = {
            "learning_rate": 1e-2,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-8,
            "force_bounds": True,
        }

        # Dynamic parameters
        self.m = 0
        self.v = 0

    def update_step(self, function: Function) -> None:

        x = self.data.get_input_data().iloc[-self.hyperparameters["population"] :].to_numpy()

        g = function.dfdx(x)
        t = self.data.get_number_of_datapoints()
        m = self.hyperparameters["beta_1"] * self.m + (1 - self.hyperparameters["beta_2"]) * g
        v = self.hyperparameters["beta_2"] * self.v + (1 - self.hyperparameters["beta_2"]) * np.power(g, 2)

        m_hat = m / (1 - np.power(self.hyperparameters["beta_1"], t))
        v_hat = v / (1 - np.power(self.hyperparameters["beta_2"], t))
        x_new = x - self.hyperparameters["learning_rate"] * m_hat / (np.sqrt(v_hat) + self.hyperparameters["epsilon"])

        # Force bounds
        if self.hyperparameters["force_bounds"]:
            x_new = self._force_bounds(x_new, function.scale_bounds)

        # Update dynamic parameters
        self.m = m
        self.v = v

        self.data.add_numpy_arrays(input=x_new, output=function(x_new))
