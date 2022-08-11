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

# Import implementation modules in separate namespaces
from . import optimization
from . import functions
from . import sampling

# Import main scripts
from .run_optimization import *
