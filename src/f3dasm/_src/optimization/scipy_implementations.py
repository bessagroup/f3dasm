#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Standard
import warnings
from collections.abc import Callable
from functools import partial
from typing import Optional

# Third-party core
import scipy.optimize

# Locals
from ..core import DataGenerator, Optimizer
from ..experimentdata import ExperimentData

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

warnings.filterwarnings(
    "ignore", message="^OptimizeWarning: Unknown solver options.*")

# =============================================================================


class ScipyOptimizer(Optimizer):
    def __init__(self, method: str,
                 bounds: Optional[scipy.optimize.Bounds] = None,
                 **hyperparameters):
        self.bounds = bounds
        self.method = method
        self.hyperparameters = hyperparameters

    def arm(self, data: ExperimentData, data_generator: DataGenerator,
            input_name: str, output_name: str):

        self.data_generator = data_generator
        self.output_name = output_name
        self.input_name = input_name

        experiment_sample = data.get_experiment_sample(data.index[-1])
        self._x0 = experiment_sample.input_data[input_name]

    def call(self, data: ExperimentData, n_iterations: Optional[int] = None,
             grad_f: Optional[Callable] = None, **kwargs) -> ExperimentData:
        history_x, history_y = [], []

        def callback(intermediate_result: scipy.optimize.OptimizeResult,
                     ) -> None:
            history_x.append(
                {self.input_name: intermediate_result.x})
            history_y.append(
                {self.output_name: intermediate_result.fun})

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
            project_dir=data.project_dir)

# =============================================================================


cg = partial(ScipyOptimizer, method='CG')
nelder_mead = partial(ScipyOptimizer, method='Nelder-Mead')
lbfgsb = partial(ScipyOptimizer, method='L-BFGS-B')

# =============================================================================
