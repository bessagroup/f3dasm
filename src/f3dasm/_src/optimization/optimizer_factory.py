"""
Module for the data generator factory.
"""
#                                                                       Modules
# =============================================================================

from typing import Any, Dict, Optional

from .._imports import try_import
from ..design.domain import Domain
from . import _OPTIMIZERS
from .optimizer import Optimizer

# Try importing f3dasm_optimize package
with try_import('f3dasm_optimize') as _imports:
    import f3dasm_optimize


#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

if _imports.is_successful():
    _OPTIMIZERS.extend(f3dasm_optimize._OPTIMIZERS)

OPTIMIZER_MAPPING: Dict[str, Optimizer] = {
    opt.__name__.lower().replace(' ', '').replace('-', '').replace('_', ''): opt for opt in _OPTIMIZERS}


OPTIMIZERS = [opt.__name__ for opt in _OPTIMIZERS]


def optimizer_factory(optimizer: str, domain: Domain, hyperparameters: Optional[Dict[str, Any]] = None) -> Optimizer:
    """Factory function for optimizers

    Parameters
    ----------

    optimizer : str
        Name of the optimizer to use
    domain : Domain
        Domain of the design space
    hyperparameters : dict, optional
        Hyperparameters for the optimizer

    Returns
    -------

    Optimizer
        Optimizer instance

    Raises
    ------

    KeyError
        If the optimizer is not found
    """

    if hyperparameters is None:
        hyperparameters = {}

    filtered_name = optimizer.lower().replace(' ', '').replace('-', '').replace('_', '')

    if filtered_name in OPTIMIZER_MAPPING:
        return OPTIMIZER_MAPPING[filtered_name](domain=domain, **hyperparameters)

    else:
        raise KeyError(f"Unknown optimizer: {optimizer}")
