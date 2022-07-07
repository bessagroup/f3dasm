import numpy as np

from f3dasm.base.optimization import Optimizer
from f3dasm.base.simulation import Function


class SGD(Optimizer):
    def set_hyperparameters(self):
        hyperparameters = {"step_size": 1e-2}

        hyperparameters.update(self.hyperparameters)

        for parameter in hyperparameters:
            self.__setattr__(parameter, hyperparameters[parameter])

    def update_step(self, function: Function) -> None:

        x = self.data.get_input_data().iloc[-self.population :].to_numpy()
        # fx = self.data.get_output_data().iloc[-self.population :].to_numpy()

        g = function.dfdx(x)

        x_new = x - self.step_size * g

        self.data.add_numpy_arrays(input=x_new, output=function.eval(x_new))


class Momentum(Optimizer):
    def init_parameters(self):
        self.m = 0

    def set_hyperparameters(self):
        hyperparameters = {"step_size": 1e-2, "beta": 0.9}

        hyperparameters.update(self.hyperparameters)

        for parameter in hyperparameters:
            self.__setattr__(parameter, hyperparameters[parameter])

    def update_step(self, function: Function) -> None:

        x = self.data.get_input_data().iloc[-self.population :].to_numpy()

        g = function.dfdx(x)

        m = self.beta * self.m + (1 - self.beta) * g
        x_new = x - self.step_size * m

        self.data.add_numpy_arrays(input=x_new, output=function.eval(x_new))
        self.m = m


class Adam(Optimizer):
    def init_parameters(self):
        self.m = 0
        self.v = 0

    def set_hyperparameters(self):
        hyperparameters = {
            "step_size": 1e-2,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-8,
        }

        hyperparameters.update(self.hyperparameters)

        for parameter in hyperparameters:
            self.__setattr__(parameter, hyperparameters[parameter])

    def update_step(self, function: Function) -> None:

        x = self.data.get_input_data().iloc[-self.population :].to_numpy()

        g = function.dfdx(x)
        t = self.data.get_number_of_datapoints()
        m = self.beta_1 * self.m + (1 - self.beta_2) * g
        v = self.beta_2 * self.v + (1 - self.beta_2) * np.power(g, 2)

        m_hat = m / (1 - np.power(self.beta_1, t))
        v_hat = v / (1 - np.power(self.beta_2, t))
        x_new = x - self.step_size * m_hat / (np.sqrt(v_hat) + self.epsilon)

        self.data.add_numpy_arrays(input=x_new, output=function.eval(x_new))

        self.m = m
        self.v = v
