"""
Module for the data generator factory.
"""
#                                                                       Modules
# =============================================================================

from typing import Any, Dict, Optional

from ..design.domain import Domain
from . import OPTIMIZERS
from .optimizer import Optimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

OPTIMIZER_MAPPING: Dict[str, Optimizer] = {
    f.__name__.lower().replace(' ', '').replace('-', '').replace('_', ''): f for f in OPTIMIZERS}


def optimizer_factory(optimizer: str, domain: Domain, hyperparameters: Optional[Dict[str, Any]] = None):

    if hyperparameters is None:
        hyperparameters = {}

    filtered_name = optimizer.lower().replace(' ', '').replace('-', '').replace('_', '')

    if filtered_name in OPTIMIZER_MAPPING:
        return OPTIMIZER_MAPPING[filtered_name](dimensionality=len(domain), **hyperparameters)

    else:
        raise KeyError(f"Unknown optimizer: {optimizer}")
