"""
Utility functions for the experimentdata module
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import warnings
from functools import wraps

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#


def number_of_updates(iterations: int, population: int):
    """Calculate number of update steps to acquire the
     correct number of iterations

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


def deprecated(version: str = "", message: str = ""):
    """
    A decorator to mark functions as deprecated.
    It will display a warning when the function is used.

    Args:
        version (str): Optional version when the method will be removed.
        message (str): Additional message to display in the warning.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warning_msg = f"{func.__name__}() is deprecated"
            if version:
                warning_msg += f" and will be removed in version {version}"
            if message:
                warning_msg += f". {message}"
            warnings.warn(
                warning_msg, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator
