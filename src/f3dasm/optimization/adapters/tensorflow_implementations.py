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

from ...base.utils import SimpelModel, convert_autograd_to_tensorflow
# Locals
from .._protocol import Function
from ..optimizer import Optimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling', 'Surya Manoj Sanu']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

class TensorflowOptimizer(Optimizer):
    def init_parameters(self):
        self.args = {}

    def update_step(self, function: Function) -> Tuple[np.ndarray, np.ndarray]:
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
        self.args["model"] = SimpelModel(
            None,
            args={
                "dim": self.data.design.get_number_of_input_parameters(),
                "x0": self.data.get_n_best_input_parameters_numpy(self.parameter.population),
                "bounds": self.data.design.get_bounds(),
            },
        )  # Build the model
        self.args["tvars"] = self.args["model"].trainable_variables
        self.args["func"] = convert_autograd_to_tensorflow(function.__call__)
