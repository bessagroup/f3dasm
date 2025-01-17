"""
Module for the data generator factory.
"""
#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from typing import Callable, Dict, List

# Local
from ..core import Block
from .numpy_implementations import random_search
from .scipy_implementations import cg, lbfgsb, nelder_mead

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def available_optimizers():
    """
    Returns a list of all available built-in optimization algorithms.

    Returns
    -------
    List[str]
        List of all available optimization algorithms
    """
    return list(get_optimizer_mapping().keys())


def get_optimizer_mapping() -> Dict[str, Block]:
    # List of available optimizers
    _OPTIMIZERS: List[Callable] = [
        cg, lbfgsb, nelder_mead, random_search]

    # Try importing f3dasm_optimize package
    try:
        from f3dasm_optimize import optimizers_extension  # NOQA
        _OPTIMIZERS.extend(optimizers_extension())
    except ImportError:
        pass

    OPTIMIZER_MAPPING: Dict[str, Block] = {
        opt.__name__.lower().replace(' ', '').replace('-', '').replace(
            '_', ''): opt for opt in _OPTIMIZERS}

    return OPTIMIZER_MAPPING


def _optimizer_factory(optimizer: str | Block, **hyperparameters
                       ) -> Block:
    """Factory function for optimizers

    Parameters
    ----------

    optimizer : str
        Name of the optimizer to use

    Returns
    -------

    Optimizer
        Optimizer instance

    Raises
    ------

    KeyError
        If the optimizer is not found
    """
    if isinstance(optimizer, Block):
        return optimizer

    elif isinstance(optimizer, str):

        filtered_name = optimizer.lower().replace(
            ' ', '').replace('-', '').replace('_', '')

        OPTIMIZER_MAPPING = get_optimizer_mapping()

        if filtered_name in OPTIMIZER_MAPPING:
            return OPTIMIZER_MAPPING[filtered_name](
                **hyperparameters)

    else:
        raise KeyError(f"Unknown optimizer: {optimizer}")
