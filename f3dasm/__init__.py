"""
F3DASM
======

This is the initalizer of the F3DASM package
"""

# Import base class in main namespace
from .base.space import *
from .base.data import *
from .base.design import *
from .base.optimization import *
from .base.samplingmethod import *
from .base.function import *
from .base.utils import *

from .base.simulation import *

# Import implementation modules in separate namespaces
from . import optimization
from . import functions
from . import sampling
from . import simulation

# Import main scripts
from .run_optimization import *

# Configuration file structure
from . import config


# Logging things

import logging

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
