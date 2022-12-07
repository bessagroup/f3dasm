#                                                                       Modules
# =============================================================================

# Standard

import pickle
from typing import Any, List

# Third-party
import autograd.numpy as np
import pandas as pd

# Locals
from ..base.data import Data
from ..base.design import DesignSpace
from ..base.space import ContinuousParameter

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


def _number_of_updates(iterations: int, population: int):
    return iterations // population + (iterations % population > 0)


def _number_of_overiterations(iterations: int, population: int) -> int:
    overiterations: int = iterations % population
    if overiterations == 0:
        return overiterations
    else:
        return population - overiterations


def calculate_mean_std(results):
    mean_y = pd.concat([d.get_output_data().cummin()
                       for d in results], axis=1).mean(axis=1)
    std_y = pd.concat([d.get_output_data().cummin()
                      for d in results], axis=1).std(axis=1)
    return mean_y, std_y
