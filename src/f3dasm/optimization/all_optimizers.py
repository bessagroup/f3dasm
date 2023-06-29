#                                                                       Modules
# =============================================================================

# Standard
from typing import List, Protocol

# Locals
from .._imports import try_import
from . import cg, lbfgsb, neldermead, optimizer, randomsearch
from .optimizer import Optimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

# List of available models
OPTIMIZERS: List[Optimizer] = []

# Core models
OPTIMIZERS.append(randomsearch.RandomSearch)
OPTIMIZERS.append(cg.CG)
OPTIMIZERS.append(lbfgsb.LBFGSB)
OPTIMIZERS.append(neldermead.NelderMead)


# Import from f3dasm_optimize package
with try_import('f3dasm_optimize') as _imports:
    import f3dasm_optimize

if _imports.is_successful():
    OPTIMIZERS.extend(f3dasm_optimize.OPTIMIZERS)
