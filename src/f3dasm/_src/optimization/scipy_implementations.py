#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Standard
import warnings
from typing import Optional, Callable
from functools import partial

# Third-party core
from scipy.optimize import minimize, OptimizeResult, Bounds

# Locals
from ..core import Block, ExperimentData
from ..datagenerator import DataGenerator
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
    def __init__(self, method: str, bounds: Optional[Bounds] = None,
                 tol: Optional[float] = None,
                 **hyperparameters):
        self.bounds = bounds
        self.method = method
        self.tol = tol
        self.hyperparameters = hyperparameters

    def arm(self, data: ExperimentData):
        input_name = data.domain.input_names[0]
        experiment_sample = data.get_experiment_sample(data.index[-1])
        self.x0 = experiment_sample.input_data[input_name]

    def call(self, data: ExperimentData, data_generator: DataGenerator,
             grad_f: Optional[Callable] = None) -> ExperimentData:
        history_x, history_y = [], []

        def callback(intermediate_result: OptimizeResult) -> None:
            history_x.append(
                {input_name: intermediate_result.x for input_name
                 in data.domain.input_names})
            history_y.append(
                {output_name: intermediate_result.fun for output_name
                 in data_generator.output_names})

        _ = minimize(
            fun=data_generator.f,
            x0=self.x0,
            method=self.method,
            jac=grad_f,
            bounds=self.bounds,
            options={**self.hyperparameters},
            callback=callback,
            tol=self.tol)

        return ExperimentData(
            domain=data.domain,
            input_data=history_x,
            output_data=history_y,
            project_dir=data.project_dir)

# =============================================================================


OPTIMIZERS = {
    'cg': partial(ScipyOptimizer, method='CG'),
    'neldermead': partial(ScipyOptimizer, method='Nelder-Mead'),
    'lbfgsb': partial(ScipyOptimizer, method='L-BFGS-B'),
}
