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
from ..core import DataGenerator, Optimizer
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


class ScipyOptimizer(Optimizer):
    """Wrapper for scipy.optimize.minimize optimization methods.

    This class provides an interface to various optimization algorithms
    from scipy.optimize.minimize, adapted to work with the f3dasm framework.

    Parameters
    ----------
    method : str
        Name of the scipy optimization method to use (e.g., 'CG', 'L-BFGS-B',
        'Nelder-Mead').
    bounds : scipy.optimize.Bounds, optional
        Bounds on variables for optimization algorithms that support them.
    **hyperparameters
        Additional keyword arguments to pass to the scipy optimizer as options.

    Attributes
    ----------
    bounds : scipy.optimize.Bounds or None
        Bounds on variables for the optimization.
    method : str
        The scipy optimization method being used.
    hyperparameters : dict
        Dictionary of hyperparameters for the optimizer.
    data_generator : DataGenerator
        The data generator used for function evaluations.
    output_name : str
        Name of the output variable to optimize.
    input_name : str
        Name of the input variable being optimized.
    """

    def __init__(
        self,
        method: str,
        bounds: Optional[scipy.optimize.Bounds] = None,
        **hyperparameters,
    ):
        self.bounds = bounds
        self.method = method
        self.hyperparameters = hyperparameters

    def arm(
        self,
        data: ExperimentData,
        data_generator: DataGenerator,
        input_name: str,
        output_name: str,
    ):
        """Prepare the optimizer with initial data and configuration.

        Parameters
        ----------
        data : ExperimentData
            The experiment data containing initial samples.
        data_generator : DataGenerator
            The data generator used for function evaluations.
        input_name : str
            Name of the input variable to optimize.
        output_name : str
            Name of the output variable to optimize.
        """
        self.data_generator = data_generator
        self.output_name = output_name
        self.input_name = input_name

        experiment_sample = data.get_experiment_sample(data.index[-1])
        self._x0 = experiment_sample.input_data[input_name]

    def call(
        self,
        data: ExperimentData,
        n_iterations: Optional[int] = None,
        grad_f: Optional[Callable] = None,
        **kwargs,
    ) -> ExperimentData:
        """Execute the optimization algorithm.

        Parameters
        ----------
        data : ExperimentData
            The experiment data to use for optimization.
        n_iterations : int, optional
            Maximum number of iterations (not used in scipy optimizers, but
            kept for interface consistency).
        grad_f : callable, optional
            Gradient function for the objective. If None, gradients will be
            estimated numerically by scipy.
        **kwargs
            Additional keyword arguments (not used, but kept for interface
            consistency).

        Returns
        -------
        ExperimentData
            New experiment data containing the optimization history with
            input-output pairs from each iteration.
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
            jac=grad_f,
            bounds=self.bounds,
            options={**self.hyperparameters},
            callback=callback,
        )

        return ExperimentData(
            domain=data.domain,
            input_data=history_x,
            output_data=history_y,
            project_dir=data._project_dir,
        )


# =============================================================================


def cg(
    bounds: Optional[scipy.optimize.Bounds] = None, **hyperparameters
) -> ScipyOptimizer:
    """Create a Conjugate Gradient optimizer.

    Uses the Conjugate Gradient (CG) algorithm from scipy.optimize.minimize.
    This is a gradient-based optimization method suitable for unconstrained
    optimization of smooth functions.

    Parameters
    ----------
    bounds : scipy.optimize.Bounds, optional
        Bounds on variables. Note that CG typically doesn't support bounds
        directly, but this parameter is kept for interface consistency.
    **hyperparameters
        Additional options to pass to the CG optimizer. Common options include:
        - maxiter : int, maximum number of iterations
        - gtol : float, gradient norm tolerance
        - norm : str, order of vector norm used in convergence checks

    Returns
    -------
    ScipyOptimizer
        Configured CG optimizer instance.

    See Also
    --------
    scipy.optimize.minimize : The underlying scipy minimize function.
    """
    return ScipyOptimizer(method="CG", bounds=bounds, **hyperparameters)


def nelder_mead(
    bounds: Optional[scipy.optimize.Bounds] = None, **hyperparameters
) -> ScipyOptimizer:
    """Create a Nelder-Mead optimizer.

    Uses the Nelder-Mead simplex algorithm from scipy.optimize.minimize.
    This is a derivative-free optimization method that works well for
    non-smooth functions but may be slower for high-dimensional problems.

    Parameters
    ----------
    bounds : scipy.optimize.Bounds, optional
        Bounds on variables. Note that standard Nelder-Mead doesn't support
        bounds, but this parameter is kept for interface consistency.
    **hyperparameters
        Additional options to pass to the Nelder-Mead optimizer. Common
        options include:
        - maxiter : int, maximum number of iterations
        - maxfev : int, maximum number of function evaluations
        - xatol : float, absolute error in parameters for convergence
        - fatol : float, absolute error in function value for convergence

    Returns
    -------
    ScipyOptimizer
        Configured Nelder-Mead optimizer instance.

    See Also
    --------
    scipy.optimize.minimize : The underlying scipy minimize function.
    """
    return ScipyOptimizer(
        method="Nelder-Mead", bounds=bounds, **hyperparameters
    )


def lbfgsb(
    bounds: Optional[scipy.optimize.Bounds] = None, **hyperparameters
) -> ScipyOptimizer:
    """Create an L-BFGS-B optimizer.

    Uses the Limited-memory BFGS with Box constraints (L-BFGS-B) algorithm
    from scipy.optimize.minimize. This is a quasi-Newton method that handles
    bound constraints efficiently and is suitable for large-scale problems.

    Parameters
    ----------
    bounds : scipy.optimize.Bounds, optional
        Bounds on variables. L-BFGS-B natively supports box constraints.
    **hyperparameters
        Additional options to pass to the L-BFGS-B optimizer. Common options:
        - maxiter : int, maximum number of iterations
        - maxfun : int, maximum number of function evaluations
        - ftol : float, tolerance for termination based on function value
        - gtol : float, tolerance for termination based on gradient norm
        - maxcor : int, maximum number of variable metric corrections

    Returns
    -------
    ScipyOptimizer
        Configured L-BFGS-B optimizer instance.

    See Also
    --------
    scipy.optimize.minimize : The underlying scipy minimize function.
    """
    return ScipyOptimizer(method="L-BFGS-B", bounds=bounds, **hyperparameters)


# =============================================================================
