"""
Module for optimization
"""
#                                                                       Modules
# =============================================================================

# Standard
from functools import partial

from ._imports import try_import

# Local
from .errors import faulty_optimizer
from .optimizer_factory import create_optimizer

with try_import() as _optuna_imports:
    from .optuna_implementations import tpesampler

with try_import() as _scipy_imports:
    from .scipy_implementations import cg, lbfgsb, nelder_mead

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


# =============================================================================

if not _scipy_imports.is_successful():
    cg = partial(faulty_optimizer, name='cg', missing_package='scipy')
    lbfgsb = partial(faulty_optimizer, name='lbfgsb', missing_package='scipy')
    nelder_mead = partial(
        faulty_optimizer, name='nelder_mead', missing_package='scipy')

if not _optuna_imports.is_successful():
    tpesampler = partial(
        faulty_optimizer, name='tpesampler', missing_package='optuna')

__all__ = [
    'create_optimizer',
    "OptunaOptimizer",
    'tpesampler',
    'cg',
    'lbfgsb',
    'nelder_mead',
]
