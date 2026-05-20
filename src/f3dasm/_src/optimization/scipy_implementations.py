#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Standard
import warnings
from collections.abc import Callable
from typing import Optional

# Third-party core
import scipy.optimize

# Locals
from ..core import Block, DataGenerator
from ..experimentdata import ExperimentData

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================

warnings.filterwarnings(
    "ignore", message="^OptimizeWarning: Unknown solver options.*"
)

# =============================================================================


class ScipyOptimizer(Block):
    """One-shot block that wraps ``scipy.optimize.minimize``.

    Unlike ask/tell optimizers, scipy runs its own inner loop, so this block
    is *not* meant to be wrapped in a :class:`LoopBlock`. Call it once and
    it returns the full optimization history appended to the input data.

    Parameters
    ----------
    method : str
        scipy optimization method name (e.g. ``'CG'``, ``'L-BFGS-B'``,
        ``'Nelder-Mead'``).
    data_generator : DataGenerator
        The data generator whose ``f`` attribute will be optimized.
    output_name : str
        Name of the output column scipy minimizes.
    input_name : str
        Name of the input column controlled by scipy.
    bounds : scipy.optimize.Bounds, optional
        Bounds on variables for methods that support them.
    grad_f : callable, optional
        Gradient function. If ``None``, scipy estimates gradients numerically.
    **hyperparameters
        Forwarded to ``scipy.optimize.minimize`` as ``options={...}``
        (e.g. ``maxiter``).

    Attributes
    ----------
    method : str
        The scipy method used.
    data_generator : DataGenerator
        The data generator whose ``f`` attribute is optimized.
    output_name : str
        Name of the output column being minimized.
    input_name : str
        Name of the input column being optimized.
    bounds : scipy.optimize.Bounds or None
        Variable bounds.
    grad_f : callable or None
        Gradient function; ``None`` means numerical gradients.
    hyperparameters : dict
        Extra options forwarded to scipy.
    """

    def __init__(
        self,
        method: str,
        data_generator: DataGenerator,
        output_name: str,
        input_name: str,
        bounds: Optional[scipy.optimize.Bounds] = None,
        grad_f: Optional[Callable] = None,
        **hyperparameters,
    ):
        self.method = method
        self.data_generator = data_generator
        self.output_name = output_name
        self.input_name = input_name
        self.bounds = bounds
        self.grad_f = grad_f
        self.hyperparameters = hyperparameters

    def arm(self, data: ExperimentData) -> None:
        """Capture the initial point (``x0``) from the last row of ``data``.

        Parameters
        ----------
        data : ExperimentData
            The experiment data providing the initial point.
        """
        experiment_sample = data.get_experiment_sample(data.index[-1])
        self._x0 = experiment_sample.input_data[self.input_name]

    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        """Run ``scipy.optimize.minimize`` and append its history to ``data``.

        Parameters
        ----------
        data : ExperimentData
            The input experiment data. The history produced by scipy is
            appended to this via :meth:`ExperimentData.__add__`.
        **kwargs : dict
            Unused; kept for Block interface consistency.

        Returns
        -------
        ExperimentData
            ``data`` concatenated with one row per scipy callback call
            (the input-output pairs scipy emits during its inner loop).
        """
        history_x, history_y = [], []

        def callback(
            intermediate_result: scipy.optimize.OptimizeResult,
        ) -> None:
            history_x.append({self.input_name: intermediate_result.x})
            history_y.append({self.output_name: intermediate_result.fun})

        _ = scipy.optimize.minimize(
            fun=self.data_generator.f,
            x0=self._x0,
            method=self.method,
            jac=self.grad_f,
            bounds=self.bounds,
            options={**self.hyperparameters},
            callback=callback,
        )

        history = ExperimentData(
            domain=data.domain,
            input_data=history_x,
            output_data=history_y,
            project_dir=data._project_dir,
        )

        return data + history


# =============================================================================


def cg(
    data_generator: DataGenerator,
    output_name: str,
    input_name: str,
    bounds: Optional[scipy.optimize.Bounds] = None,
    grad_f: Optional[Callable] = None,
    **hyperparameters,
) -> ScipyOptimizer:
    """Create a Conjugate Gradient block.

    Parameters
    ----------
    data_generator : DataGenerator
        The data generator whose ``f`` attribute will be optimized.
    output_name : str
        Name of the output column to minimize.
    input_name : str
        Name of the input column controlled by CG.
    bounds : scipy.optimize.Bounds, optional
        Bounds on variables (CG does not natively support bounds; kept for
        interface consistency).
    grad_f : callable, optional
        Gradient function. If ``None``, gradients are estimated numerically.
    **hyperparameters
        Options forwarded to the CG optimizer, such as ``maxiter``, ``gtol``,
        ``norm``.

    Returns
    -------
    ScipyOptimizer
        Configured CG block.

    See Also
    --------
    scipy.optimize.minimize
    """
    return ScipyOptimizer(
        method="CG",
        data_generator=data_generator,
        output_name=output_name,
        input_name=input_name,
        bounds=bounds,
        grad_f=grad_f,
        **hyperparameters,
    )


def nelder_mead(
    data_generator: DataGenerator,
    output_name: str,
    input_name: str,
    bounds: Optional[scipy.optimize.Bounds] = None,
    grad_f: Optional[Callable] = None,
    **hyperparameters,
) -> ScipyOptimizer:
    """Create a Nelder-Mead block.

    Parameters
    ----------
    data_generator : DataGenerator
        The data generator whose ``f`` attribute will be optimized.
    output_name : str
        Name of the output column to minimize.
    input_name : str
        Name of the input column controlled by Nelder-Mead.
    bounds : scipy.optimize.Bounds, optional
        Bounds on variables (standard Nelder-Mead does not support them).
    grad_f : callable, optional
        Gradient function (unused by Nelder-Mead; kept for interface
        consistency).
    **hyperparameters
        Options forwarded to the Nelder-Mead optimizer, such as ``maxiter``,
        ``maxfev``, ``xatol``, ``fatol``.

    Returns
    -------
    ScipyOptimizer
        Configured Nelder-Mead block.

    See Also
    --------
    scipy.optimize.minimize
    """
    return ScipyOptimizer(
        method="Nelder-Mead",
        data_generator=data_generator,
        output_name=output_name,
        input_name=input_name,
        bounds=bounds,
        grad_f=grad_f,
        **hyperparameters,
    )


def lbfgsb(
    data_generator: DataGenerator,
    output_name: str,
    input_name: str,
    bounds: Optional[scipy.optimize.Bounds] = None,
    grad_f: Optional[Callable] = None,
    **hyperparameters,
) -> ScipyOptimizer:
    """Create an L-BFGS-B block.

    Parameters
    ----------
    data_generator : DataGenerator
        The data generator whose ``f`` attribute will be optimized.
    output_name : str
        Name of the output column to minimize.
    input_name : str
        Name of the input column controlled by L-BFGS-B.
    bounds : scipy.optimize.Bounds, optional
        Bounds on variables; L-BFGS-B supports box constraints natively.
    grad_f : callable, optional
        Gradient function. If ``None``, gradients are estimated numerically.
    **hyperparameters
        Options forwarded to the L-BFGS-B optimizer, such as ``maxiter``,
        ``maxfun``, ``ftol``, ``gtol``, ``maxcor``.

    Returns
    -------
    ScipyOptimizer
        Configured L-BFGS-B block.

    See Also
    --------
    scipy.optimize.minimize
    """
    return ScipyOptimizer(
        method="L-BFGS-B",
        data_generator=data_generator,
        output_name=output_name,
        input_name=input_name,
        bounds=bounds,
        grad_f=grad_f,
        **hyperparameters,
    )


# =============================================================================
