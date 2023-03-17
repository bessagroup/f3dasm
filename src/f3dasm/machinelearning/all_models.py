#                                                                       Modules
# =============================================================================

# Standard
from typing import List

# Locals
from . import linear_regression, passthrough_model
from .model import Model

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

# List of available models
MODELS: List[Model] = []

# Core models

# Extension samplers
if passthrough_model._imports.is_successful():
    MODELS.append(passthrough_model.PassthroughModel)

if linear_regression._imports.is_successful():
    MODELS.append(linear_regression.LinearRegression)
