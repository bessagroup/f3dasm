"""
F3DASM
======

This is the initalizer of the F3DASM package
"""

#                                                                       Modules
# =============================================================================

# Standard
import logging
from pathlib import Path

# Locals
from f3dasm import (data, functions, machinelearning, optimization, sampling,
                    simulation)

from ._show_versions import show_versions
from .base.utils import *
from .design.design import *
from .design.experimentdata import *
from .design.parameter import *
from .functions.function import *
from .optimization.optimizer import Optimizer
from .run_optimization import (OptimizationResult, run_multiple_realizations,
                               run_optimization)
from .sampling.sampler import Sampler
from .sampling.utils import *

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'

# version
here = Path(__file__).absolute().parent
with open(here.joinpath("VERSION"), "r") as f:
    __version__ = f.read()

# =============================================================================
#
# =============================================================================


# Logging things
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
