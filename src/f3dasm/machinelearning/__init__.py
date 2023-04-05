#                                                                       Modules
# =============================================================================

# Standard
import sys
from itertools import chain
from os import path
from typing import TYPE_CHECKING

# Local
from .._imports import _IntegrationModule

if TYPE_CHECKING:
    from .linear_regression import LinearRegression
    from .model import Model
    from .passthrough_model import PassthroughModel
    from .gpr import (
        Sogpr, 
        Sogpr_Parameters,
        Cokgj,
        Cokgj_Parameters,
        MultitaskGPR,
        MultitaskGPR_Parameters,
        )
    from .acquisition_functions import (
        UpperConfidenceBound,
        ExpectedImprovement,
        VFUpperConfidenceBound,
    )

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

_import_structure: dict = {
    "utils": ["find_model", "create_model_from_json", "create_model_from_dict", "MeanSquaredError"],
    "model": ["Model"],
    "linear_regression": ["LinearRegression"],
    "passthrough_model": ["PassthroughModel"],
    "evaluator": ["Evaluator"],
    "all_models": ["MODELS"],
    "loss_functions": ["MeanSquaredError"],
    "gpr": ["Sogpr", "Sogpr_parameters", "Cokgj", "Cokgj_Parameters",
        "MultitaskGPR", "MultitaskGPR_Parameters"],
    "acquisition_functions": ["UpperConfidenceBound", 
                              "ExpectedImprovement", "VFUpperConfidenceBound"],
}

if not TYPE_CHECKING:
    class _LocalIntegrationModule(_IntegrationModule):
        __file__ = globals()["__file__"]
        __path__ = [path.dirname(__file__)]
        __all__ = list(chain.from_iterable(_import_structure.values()))
        _import_structure = _import_structure

    sys.modules[__name__] = _LocalIntegrationModule(__name__)
