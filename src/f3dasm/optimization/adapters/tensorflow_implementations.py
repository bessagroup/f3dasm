#                                                                       Modules
# =============================================================================


# Standard
from typing import Callable, Tuple

# Third-party core
import autograd.core
import autograd.numpy as np
import pandas as pd
from autograd import elementwise_grad as egrad

# Locals
from ..._imports import try_import
from ...functions import Function
from ..optimizer import Optimizer

# Third-party extension
with try_import('optimization') as _imports:
    import tensorflow as tf
    from keras import Model


#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling', 'Surya Manoj Sanu']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

if not _imports.is_successful():
    Model = object  # NOQA


class TensorflowOptimizer(Optimizer):
    @staticmethod
    def _check_imports():
        _imports.check()

    def init_parameters(self):
        self.args = {}

    def update_step(self, function: Function) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(function, Function):
            logits = self.data.get_input_data().iloc[-1].to_numpy()
            x, y = function.tf_apply_gradients(weights=logits, optimizer=self.algorithm)
        else:
            with tf.GradientTape() as tape:
                tape.watch(self.args["tvars"])
                logits = 0.0 + tf.cast(self.args["model"](None), tf.float64)
                loss = self.args["func"](tf.reshape(
                    logits, (self.data.design.get_number_of_input_parameters())))

            grads = tape.gradient(loss, self.args["tvars"])
            self.algorithm.apply_gradients(zip(grads, self.args["tvars"]))

            x = logits.numpy().copy()
            y = loss.numpy().copy()

        return x, y

    def _construct_model(self, function: Function):
        self.args["model"] = _SimpelModel(
            None,
            args={
                "dim": self.data.design.get_number_of_input_parameters(),
                "x0": self.data.get_n_best_input_parameters_numpy(self.parameter.population),
                "bounds": self.data.design.get_bounds(),
            },
        )  # Build the model
        self.args["tvars"] = self.args["model"].trainable_variables

        self.args["func"] = _convert_autograd_to_tensorflow(function.__call__)


def _convert_autograd_to_tensorflow(func: Callable):
    """Convert autograd function to tensorflow function

    Parameters
    ----------
    func
        callable function to convert

    Returns
    -------
        wrapper to convert autograd function to tensorflow function
    """
    @tf.custom_gradient
    def wrapper(x, *args, **kwargs):
        vjp, ans = autograd.core.make_vjp(func, x.numpy())

        def first_grad(dy):
            @tf.custom_gradient
            def jacobian(a):
                vjp2, ans2 = autograd.core.make_vjp(egrad(func), a.numpy())
                return ans2, vjp2  # hessian

            return dy * jacobian(x)

        return ans, first_grad

    return wrapper


class _Model(Model):
    def __init__(self, seed=None, args=None):
        super().__init__()
        self.seed = seed
        self.env = args


class _SimpelModel(_Model):
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
