#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Standard
import warnings
from functools import partial
from typing import Callable, Optional

# Third-party core
import scipy.optimize

# Locals
from ..core import Block
from ..datagenerator import DataGenerator
from ..experimentdata import ExperimentData

# from scipy.optimize import Bounds, OptimizeResult, minimize


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


class ScipyOptimizer(Block):
    def __init__(self, method: str,
                 bounds: Optional[scipy.optimize.Bounds] = None,
                 **hyperparameters):
        self.bounds = bounds
        self.method = method
        self.hyperparameters = hyperparameters

    def arm(self, data: ExperimentData, data_generator: DataGenerator,
            output_name: str):

        self.data_generator = data_generator

        self.output_name = output_name
        input_name = data.domain.input_names[0]
        experiment_sample = data.get_experiment_sample(data.index[-1])
        self._x0 = experiment_sample.input_data[input_name]

    def call(self, data: ExperimentData, n_iterations: Optional[int] = None,
             grad_f: Optional[Callable] = None, **kwargs) -> ExperimentData:
        history_x, history_y = [], []

        def callback(intermediate_result: scipy.optimize.OptimizeResult,
                     ) -> None:
            history_x.append(
                {input_name: intermediate_result.x for input_name
                 in data.domain.input_names})
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
