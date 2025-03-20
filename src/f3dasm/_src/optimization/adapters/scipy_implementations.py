#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Standard
import warnings
from copy import deepcopy

# Third-party core
import autograd.numpy as np
from scipy.optimize import minimize

# Locals
from ...core import Block, DataGenerator, ExperimentData
from ...experimentsample import ExperimentSample

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


class ScipyOptimizer(Block):
    require_gradients: bool = False
    type: str = 'scipy'

    def __init__(self, algorithm, **hyperparameters):
        self.algorithm = algorithm
        self.hyperparameters = hyperparameters

    def _callback(self, xk: np.ndarray, *args, **kwargs) -> None:
        self.data.add_experiments(
            type(self.data)(domain=self.data.domain,
                            input_data=np.atleast_2d(xk),
                            project_dir=self.data.project_dir),
            in_place=True
        )

    def call(self, data: ExperimentData, **kwargs):
        """Update step function"""
        raise ValueError(
            'Scipy optimizers don\'t have an update steps. \
                 Multiple iterations are directly called \
                    througout scipy.minimize.')

    def run_algorithm(self, data_generator: DataGenerator, iterations: int):
        """Run the algorithm for a number of iterations

        Parameters
        ----------
        iterations
            number of iterations
        """

        def fun(x):
            x_ = ExperimentSample.from_numpy(input_array=x,
                                             domain=self.data.domain)
            sample = data_generator.execute(experiment_sample=x_)
            _, y = sample.to_numpy()
            return float(y)

        if not hasattr(data_generator, 'dfdx'):
            data_generator.dfdx = None

        self.hyperparameters['maxiter'] = iterations

        minimize(
            fun=fun,
            method=self.algorithm,
            jac=data_generator.dfdx,
            x0=self.data.get_n_best_output(1).to_numpy()[0].ravel(),
            callback=self._callback,
            options=self.hyperparameters,
            bounds=self.data.domain.get_bounds(),
            tol=0.0,
        )

    def _iterate(self, data: ExperimentData, data_generator: DataGenerator,
                 iterations: int,
                 kwargs: dict, overwrite: bool):
        """Internal represenation of the iteration process for scipy-minimize
        optimizers.

        Parameters
        ----------
        data_generator : DataGenerator
            DataGenerator object
        iterations : int
            number of iterations
        kwargs : Dict[str, Any]
            any additional keyword arguments that will be passed to
            the DataGenerator
        overwrite: bool
            If True, the optimizer will overwrite the current data.
        """
        self.data = data
        n_data_before_iterate = deepcopy(len(data))
        if len(data) < 1:
            raise ValueError(
                f'There are {len(data)} datapoints available, \
                        need 1 for initial \
                            population!'
            )

        self.run_algorithm(data_generator=data_generator,
                           iterations=iterations)

        new_samples = self.data.select(self.data.index[1:])

        data_generator.arm(data=new_samples)

        new_samples = data_generator.call(data=new_samples, mode='sequential')

        if overwrite:
            self.data.add_experiments(
                data.select([self.data.index[-1]]), in_place=True)

        elif not overwrite:
            # If x_new is empty, repeat best x0 to fill up total iteration
            if len(self.data) == n_data_before_iterate:
                repeated_sample = self.data.get_n_best_output(
                    n_samples=1)

                for repetition in range(iterations):
                    self.data.add_experiments(repeated_sample, in_place=True)

            # Repeat last iteration to fill up total iteration
            if len(self.data) < n_data_before_iterate + iterations:
                last_design = self.data.get_experiment_sample(len(self.data)-1)

                while len(self.data) < n_data_before_iterate + iterations:
                    self.data.add_experiments(last_design, in_place=True)

        self.data = data_generator.call(data=self.data, mode='sequential')

        return self.data
