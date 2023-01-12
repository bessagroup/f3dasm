#                                                                       Modules
# =============================================================================

# Standard

import pickle
from typing import Any, List

# Third-party
import autograd
import autograd.core
import autograd.numpy as np
import pandas as pd
import tensorflow as tf
from autograd import elementwise_grad as egrad

# Locals
from .data import Data
from .design import DesignSpace
from .space import ContinuousParameter

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def make_nd_continuous_design(bounds: np.ndarray, dimensionality: int) -> DesignSpace:
    """Helper function to make an continuous design space with a single-objective continuous output

    Parameters
    ----------
    bounds
        lower and upper bounds of every dimension
    dimensionality
        number of dimensions

    Returns
    -------
        continuous, single-objective designspace
    """
    input_space, output_space = [], []
    for dim in range(dimensionality):
        input_space.append(ContinuousParameter(
            name=f"x{dim}", lower_bound=bounds[dim, 0], upper_bound=bounds[dim, 1]))

    output_space.append(ContinuousParameter(name="y"))

    return DesignSpace(input_space=input_space, output_space=output_space)


def _from_data_to_numpy_array_benchmarkfunction(
    data: Data,
) -> np.ndarray:
    """Check if doe is in right format"""
    if not data.design.is_single_objective_continuous():
        raise TypeError(
            "All inputs and outputs need to be continuous parameters and output single objective")

    return data.get_input_data().to_numpy()


def _scale_vector(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Scale a vector x to a given scale"""
    return (scale[:, 1] - scale[:, 0]) * x + scale[:, 0]


def _descale_vector(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Inverse of the _scale_vector() function"""
    return (x - scale[:, 0]) / (scale[:, 1] - scale[:, 0])


def _rotate_vector(x: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Rotate a vector with a matrix"""
    return x.dot(rotation_matrix)


def find_class(module, query: str):
    """Find a class from a string

    Parameters
    ----------
    module
        (sub)module to be searching
    query
        string to search for

    Returns
    -------
        class
    """
    return getattr(module, query)


def read_pickle(name: str) -> Any:
    """Read a pickled object to memory

    Parameters
    ----------
    name
        name of file without extension .obj

    Returns
    -------
        object
    """
    with open(f"{name}.obj", "rb") as f:
        obj = pickle.load(f)
    return obj


def write_pickle(name: str, obj: Any):
    """Write an object to a file with pickle

    Parameters
    ----------
    name
        name of file to write without file exentions .obj
    obj
        object to store
    """
    with open(f"{name}.obj", "wb") as f:
        pickle.dump(obj, f)

def calculate_mean_std(results):  # OptimizationResult
    mean_y = pd.concat([d.get_output_data().cummin()
                       for d in results.data], axis=1).mean(axis=1)
    std_y = pd.concat([d.get_output_data().cummin()
                      for d in results.data], axis=1).std(axis=1)
    return mean_y, std_y


# FUNCTIONS FOR CALCULATING THE GRADIENT

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

def get_reshaped_array_from_list_of_arrays(flat_array: np.ndarray, list_of_arrays: List[np.ndarray]) -> List[np.ndarray]:
    total_array = []
    index = 0
    for mimic_array in list_of_arrays:
        number_of_values = np.product(mimic_array.shape)
        current_array = np.array(flat_array[index:index+number_of_values])

        if number_of_values > 1:
            current_array = current_array.reshape(-1, 1)  # Make 2D array

        total_array.append(current_array)
        index += number_of_values

    return total_array

def get_flat_array_from_list_of_arrays(list_of_arrays: List[np.ndarray]) -> List[np.ndarray]: # technically not a np array input!
    return np.concatenate([np.atleast_2d(array) for array in list_of_arrays])