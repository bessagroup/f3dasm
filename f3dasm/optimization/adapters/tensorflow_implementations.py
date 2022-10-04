from dataclasses import dataclass

import autograd.numpy as np
import tensorflow as tf

from ...base.function import Function
from ...base.optimization import Optimizer, OptimizerParameters

import autograd
import autograd.core
import autograd.numpy as np
from autograd import elementwise_grad as egrad


def convert_autograd_to_tensorflow(func):  # S:func is completely written in numpy autograd
    @tf.custom_gradient
    def wrapper(x):
        vjp, ans = autograd.core.make_vjp(func, x.numpy())

        def first_grad(dy):
            @tf.custom_gradient
            def jacobian(a):
                vjp2, ans2 = autograd.core.make_vjp(egrad(func), a.numpy())
                return ans2, vjp2  # hessian

            return dy * jacobian(x)

        return ans, first_grad

    return wrapper


class Model(tf.keras.Model):
    def __init__(self, seed=None, args=None):
        super().__init__()
        self.seed = seed
        self.env = args


class SimpelModel(Model):
    """
    The class for performing optimization in the input space of the functions.
    """

    def __init__(self, seed=None, args=None):
        super().__init__(seed)
        self.z = tf.Variable(args["x0"], trainable=True, dtype=tf.float32)  # S:ADDED

    def call(self, inputs=None):
        return self.z


class TensorflowOptimizer(Optimizer):
    def init_parameters(self):
        self.args = {}

    def update_step(self, function: Function) -> None:
        with tf.GradientTape() as tape:
            tape.watch(self.args["tvars"])
            logits = 0.0 + tf.cast(self.args["model"](None), tf.float64)
            loss = self.args["func"](tf.reshape(logits, (function.dimensionality)))

        grads = tape.gradient(loss, self.args["tvars"])
        self.algorithm.apply_gradients(zip(grads, self.args["tvars"]))

        self.data.add_numpy_arrays(input=logits.numpy().copy(), output=loss.numpy().copy())

    def iterate(self, iterations: int, function: Function) -> None:
        self.args["model"] = SimpelModel(
            None,
            args={
                "dim": function.dimensionality,
                "x0": self.data.get_n_best_input_parameters_numpy(self.parameter.population),
            },
        )  # Build the model
        self.args["tvars"] = self.args["model"].trainable_variables
        self.args["func"] = convert_autograd_to_tensorflow(function.__call__)

        self._check_number_of_datapoints()

        for _ in range(self._number_of_updates(iterations)):
            self.update_step(function)
