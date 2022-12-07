#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass
from typing import Any

# Third-party
import GPy
import GPyOpt

# Locals
from ..base.function import Function
from ..base.optimization import Optimizer, OptimizerParameters

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@dataclass
class BayesianOptimization_Parameters(OptimizerParameters):
    """Hyperparameters for BayesianOptimization optimizer"""

    model: Any = None
    space: Any = None
    acquisition: Any = None
    evaluator: Any = None
    de_duplication: Any = None


class BayesianOptimization(Optimizer):
    """Bayesian Optimization implementation from the GPyOPt library"""

    parameter: BayesianOptimization_Parameters = BayesianOptimization_Parameters()

    def init_parameters(self):
        domain = [
            {
                "name": f"var_{index}",
                "type": "continuous",
                "domain": (parameter.lower_bound, parameter.upper_bound),
            }
            for index, parameter in enumerate(self.data.design.get_continuous_input_parameters())
        ]

        kernel = GPy.kern.RBF(
            input_dim=self.data.design.get_number_of_input_parameters())

        model = GPyOpt.models.gpmodel.GPModel(
            kernel=kernel,
            max_iters=1000,
            optimize_restarts=5,
            sparse=False,
            verbose=False,
        )

        space = GPyOpt.Design_space(space=domain)
        acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)
        acquisition = GPyOpt.acquisitions.AcquisitionEI(
            model, space, acquisition_optimizer)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

        # Default hyperparamaters
        options = {
            "model": model,
            "space": space,
            "acquisition": acquisition,
            "evaluator": evaluator,
            "de_duplication": True,
        }

        self.parameter = BayesianOptimization_Parameters(**options)

    def set_algorithm(self):
        self.algorithm = GPyOpt.methods.ModularBayesianOptimization(
            model=self.parameter.model,
            space=self.parameter.space,
            objective=None,
            acquisition=self.parameter.acquisition,
            evaluator=self.parameter.evaluator,
            X_init=None,
            Y_init=None,
            de_duplication=self.parameter.de_duplication,
        )

    def update_step(self, function: Function):

        self.algorithm.objective = GPyOpt.core.task.SingleObjective(
            function.__call__)
        self.algorithm.X = self.data.get_input_data().to_numpy()
        self.algorithm.Y = self.data.get_output_data().to_numpy()

        x_new = self.algorithm.suggest_next_locations()

        self.data.add_numpy_arrays(input=x_new, output=function(x_new))
