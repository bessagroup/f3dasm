"""
F3DASM
======

This is the initalizer of the F3DASM package
"""

__version__ = '0.2.97'

#                                                                       Modules
# =============================================================================

# Standard
import logging

# Locals
from . import functions, machinelearning, optimization, sampling, simulation
from ._show_versions import show_versions
from .base.evaluator import Evaluator
from .base.function import *
from .base.utils import *
from .design.design import *
from .design.experimentdata import *
from .design.parameter import *
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
