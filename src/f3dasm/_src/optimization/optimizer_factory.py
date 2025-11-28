"""
Module for the data generator factory.
"""
#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Local
from ..core import Block
from ._imports import try_import

with try_import() as _optuna_imports:
    from .optuna_implementations import tpesampler

with try_import() as _scipy_imports:
    from .scipy_implementations import cg, lbfgsb, nelder_mead

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================

OPTIMIZERS = {}

if _scipy_imports.is_successful():
    OPTIMIZERS["cg"] = cg
    OPTIMIZERS["neldermead"] = nelder_mead
    OPTIMIZERS["lbfgsb"] = lbfgsb

if _optuna_imports.is_successful():
    OPTIMIZERS["tpesampler"] = tpesampler


# =============================================================================
def create_optimizer(optimizer: str, **hyperparameters) -> Block:
    """
    Create a optimizer block from one of the built-in optimizers.

    Parameters
    ----------
    optimizer : str | Block
        name of the built-in optimizer. This can be a string with the name of
        the optimizer, a Block object (this will just by-pass the function).
    **hyperparameters
        Additional keyword arguments passed when initializing the optimizer

    Returns
    -------
    Block
        Block object of the optimizer

    Raises
    ------
    KeyError
        If the built-in optimizer name is not recognized.
    TypeError
        If the given type is not recognized.
    """

    if isinstance(optimizer, str):
        filtered_name = (
            optimizer.lower()
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
        )

        if filtered_name in OPTIMIZERS:
            return OPTIMIZERS[filtered_name](**hyperparameters)
        else:
            raise KeyError(f"Unknown optimizer name: {optimizer}")

    else:
        raise TypeError(f"Unknown optimizer type: {type(optimizer)}")
