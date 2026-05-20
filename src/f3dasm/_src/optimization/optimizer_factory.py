"""Factory for built-in optimizer blocks."""
#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Third-party
from hydra.utils import instantiate
from omegaconf import DictConfig

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
def create_optimizer(optimizer: str | DictConfig, **hyperparameters) -> Block:
    """Create an optimizer block from one of the built-in optimizers.

    The returned block is always a single step of the optimizer's work,
    never a full loop:

    - For ask/tell optimizers (e.g. ``"tpesampler"``) the block is the
      per-iteration update step and must be chained with a data generator
      and wrapped in a :class:`LoopBlock`::

          tpe = create_optimizer("tpesampler", output_name="y")
          step = (tpe >> f).loop(50)
          data = step.call(data)

    - For scipy optimizers (e.g. ``"cg"``, ``"lbfgsb"``, ``"neldermead"``)
      the block is a one-shot optimizer that runs scipy's own inner loop
      on a single call; pass ``maxiter`` via ``hyperparameters``, and do
      *not* wrap it in a :class:`LoopBlock`::

          step = create_optimizer(
              "cg", data_generator=f, output_name="y",
              input_name="x", maxiter=50,
          )
          data = step.call(data)

    Parameters
    ----------
    optimizer : str | DictConfig
        Name of the built-in optimizer, or a Hydra ``DictConfig`` that
        instantiates an optimizer block.
    **hyperparameters
        Forwarded to the underlying factory function.

    Returns
    -------
    Block
        Configured optimizer block.

    Raises
    ------
    KeyError
        If the built-in optimizer name is not recognized.
    TypeError
        If ``optimizer`` is not a ``str`` or ``DictConfig``.
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

    elif isinstance(optimizer, DictConfig):
        return instantiate(optimizer)

    else:
        raise TypeError(f"Unknown optimizer type: {type(optimizer)}")
