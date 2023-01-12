"""
F3DASM
======

This is the initalizer of the F3DASM package
"""

__version__ = '0.2.94'

#                                                                       Modules
# =============================================================================

# Standard
import logging

# Locals
from . import (config, functions, machinelearning, optimization, sampling,
               simulation)
from ._show_versions import show_versions
from .base.data import *
from .base.design import *
from .base.function import *
from .base.metaoptimizer import *
from .base.simulation import *
from .base.space import *
from .base.utils import *
from .optimization.optimizer import *
from .run_optimization import *
from .sampling.sampler import *

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


# Logging things
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
