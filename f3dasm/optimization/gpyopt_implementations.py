import GPy
import GPyOpt

from ..base.optimization import Optimizer
from ..base.function import Function


class BayesianOptimization(Optimizer):
    """Bayesian Optimization implementation from the GPyOPt library"""

    def init_parameters(self):
        domain = [
            {
                "name": f"var_{index}",
                "type": "continuous",
                "domain": (parameter.lower_bound, parameter.upper_bound),
            }
            for index, parameter in enumerate(self.data.designspace.get_continuous_input_parameters())
        ]

        kernel = GPy.kern.RBF(input_dim=self.data.designspace.get_number_of_input_parameters())

        model = GPyOpt.models.gpmodel.GPModel(
            kernel=kernel,
            max_iters=1000,
            optimize_restarts=5,
            sparse=False,
            verbose=False,
        )

        space = GPyOpt.Design_space(space=domain)
        acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)
        acquisition = GPyOpt.acquisitions.AcquisitionEI(model, space, acquisition_optimizer)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

        # Default hyperparamaters
        self.defaults = {
            "model": model,
            "space": space,
            "acquisition": acquisition,
            "evaluator": evaluator,
            "de_duplication": True,
        }

    def set_algorithm(self):
        self.algorithm = GPyOpt.methods.ModularBayesianOptimization(
            model=self.hyperparameters["model"],
            space=self.hyperparameters["space"],
            objective=None,
            acquisition=self.hyperparameters["acquisition"],
            evaluator=self.hyperparameters["evaluator"],
            X_init=None,
            Y_init=None,
            de_duplication=self.hyperparameters["de_duplication"],
        )

    def update_step(self, function: Function) -> None:

        self.algorithm.objective = GPyOpt.core.task.SingleObjective(function.__call__)
        self.algorithm.X = self.data.get_input_data().to_numpy()
        self.algorithm.Y = self.data.get_output_data().to_numpy()

        x_new = self.algorithm.suggest_next_locations()

        self.data.add_numpy_arrays(input=x_new, output=function.__call__(x_new))
