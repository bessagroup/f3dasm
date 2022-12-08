"""
F3DASM
======

This is the initalizer of the F3DASM package
"""
__version__ = '0.2.91'

import logging

# Configuration file structure
# Import implementation modules in separate namespaces
from . import config, functions, optimization, sampling, simulation
from ._show_versions import show_versions
from .base.data import *
from .base.design import *
from .base.function import *
from .base.metaoptimizer import *
from .base.optimization import *
from .base.samplingmethod import *
from .base.simulation import *
# Import base class in main namespace
from .base.space import *
from .base.utils import *
# Import main scripts
from .run_optimization import *

# Logging things
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
