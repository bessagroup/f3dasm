#                                                                       Modules
# =============================================================================


# Standard
from typing import Tuple

# Third-party
import autograd
import autograd.core
import autograd.numpy as np
import tensorflow as tf
from autograd import elementwise_grad as egrad

# Locals
from ...base.function import Function
from ...base.optimization import Optimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling', 'Surya Manoj Sanu']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


# S:func is completely written in numpy autograd
def convert_autograd_to_tensorflow(func):
    """Convert autograd function to tensorflow funciton

    :param func: function
    :return: wrapper
    """

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
        self.z = tf.Variable(
            args["x0"],
            trainable=True,
            dtype=tf.float32,
            constraint=lambda x: tf.clip_by_value(
                x,
                clip_value_min=args["bounds"][:, 0],
                clip_value_max=args["bounds"][:, 1],
            ),
        )  # S:ADDED

    def call(self, inputs=None):
        return self.z


class TensorflowOptimizer(Optimizer):
    def init_parameters(self):
        self.args = {}

    def update_step(self, function: Function) -> Tuple[np.ndarray, np.ndarray]:
        with tf.GradientTape() as tape:
            tape.watch(self.args["tvars"])
            logits = 0.0 + tf.cast(self.args["model"](None), tf.float64)
            loss = self.args["func"](tf.reshape(
                logits, (function.dimensionality)))

        grads = tape.gradient(loss, self.args["tvars"])
        self.algorithm.apply_gradients(zip(grads, self.args["tvars"]))

        x = logits.numpy().copy()
        y = loss.numpy().copy()
        return x, y

    def _construct_model(self, function: Function):
        self.args["model"] = SimpelModel(
            None,
            args={
                "dim": function.dimensionality,
                "x0": self.data.get_n_best_input_parameters_numpy(self.parameter.population),
                "bounds": self.data.design.get_bounds(),
            },
        )  # Build the model
        self.args["tvars"] = self.args["model"].trainable_variables
        self.args["func"] = convert_autograd_to_tensorflow(function.__call__)
