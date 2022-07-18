import GPy
import GPyOpt

from f3dasm.base.optimization import Optimizer
from f3dasm.base.simulation import Function


class BayesianOptimization(Optimizer):
    def init_parameters(self):

        # Default hyperparamaters
        domain = [
            {
                "name": f"var_{index}",
                "type": "continuous",
                "domain": (parameter.lower_bound, parameter.upper_bound),
                "dimensionality": 1,
            }
            for index, parameter in enumerate(
                self.data.designspace.get_continuous_parameters()
            )
        ]

        kernel = GPy.kern.RBF(
            input_dim=self.data.designspace.get_number_of_input_parameters()
        )

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
            model, space, acquisition_optimizer
        )
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

        self.defaults = {
            "model": model,
            "space": space,
            "acquisition": acquisition,
            "evaluator": evaluator,
            "de_duplication": True,
        }

        # Dynamic parameters

    def update_step(self, function: Function) -> None:

        x = self.data.get_input_data().to_numpy()
        y = self.data.get_output_data().to_numpy().reshape(-1, 1)

        bo = GPyOpt.methods.ModularBayesianOptimization(
            model=self.hyperparameters["model"],
            space=self.hyperparameters["space"],
            objective=GPyOpt.core.task.SingleObjective(function.eval),
            acquisition=self.hyperparameters["acquisition"],
            evaluator=self.hyperparameters["evaluator"],
            X_init=x,
            Y_init=y,
            de_duplication=self.hyperparameters["de_duplication"],
        )

        x_new = bo.suggest_next_locations()

        self.data.add_numpy_arrays(input=x_new, output=function.eval(x_new))
