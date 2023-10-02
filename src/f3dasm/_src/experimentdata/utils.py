"""
Utility functions for the experimentdata module
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from pathlib import Path
from typing import Union

# Third-party
import numpy as np
import pandas as pd

# Local
from ._data import _Data

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#

DataTypes = Union[pd.DataFrame, np.ndarray, Path, str, _Data]
DOMAIN_SUFFIX = "_domain"
INPUT_DATA_SUFFIX = "_input"
OUTPUT_DATA_SUFFIX = "_output"
JOBS_SUFFIX = "_jobs"


def number_of_updates(iterations: int, population: int):
    """Calculate number of update steps to acquire the correct number of iterations

    Parameters
    ----------
    iterations
        number of desired iteration steps
    population
        the population size of the optimizer

    Returns
    -------
        number of consecutive update steps
    """
    return iterations // population + (iterations % population > 0)


def number_of_overiterations(iterations: int, population: int) -> int:
    """Calculate the number of iterations that are over the iteration limit

    Parameters
    ----------
    iterations
        number of desired iteration steos
    population
        the population size of the optimizer

    Returns
    -------
        number of iterations that are over the limit
    """
    overiterations: int = iterations % population
    if overiterations == 0:
        return overiterations
    else:
        return population - overiterations
