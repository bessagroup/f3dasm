#                                                                       Modules
# =============================================================================
from __future__ import annotations

import warnings
# Standard
from copy import deepcopy
from typing import Callable, Protocol

# Third-party core
import autograd.numpy as np
from scipy.optimize import minimize

# Locals
from ...core import DataGenerator, Optimizer
from ...experimentsample import ExperimentSample
from .._protocol import ExperimentData

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


class Sampler(Protocol):
    ...


class ScipyOptimizer(Optimizer):
    require_gradients: bool = False
    type: str = 'scipy'

    def __init__(self, algorithm_cls, **hyperparameters):
        self.algorithm_cls = algorithm_cls
        self.hyperparameters = hyperparameters

    def arm(self, data: ExperimentData, data_generator: DataGenerator):
        self.data = data
        self.algorithm = self.algorithm_cls
        self.data_generator = data_generator

    def _callback(self, xk: np.ndarray, *args, **kwargs) -> None:
        self.data.add_experiments(
            ExperimentSample.from_numpy(input_array=xk,
                                        domain=self.data.domain))

    def update_step(self):
        """Update step function"""
        raise ValueError(
            'Scipy optimizers don\'t have an update steps. \
                 Multiple iterations are directly called \
                    througout scipy.minimize.')

    def run_algorithm(self, iterations: int):
        """Run the algorithm for a number of iterations

        Parameters
        ----------
        iterations
            number of iterations
        """

        def fun(x):
            x_ = ExperimentSample.from_numpy(input_array=x,
                                             domain=self.data.domain)
            sample: ExperimentSample = self.data_generator._run(
                x_, domain=self.data.domain)
            _, y = sample.to_numpy()
            return float(y)

        if not hasattr(self.data_generator, 'dfdx'):
            self.data_generator.dfdx = None

        self.hyperparameters['maxiter'] = iterations

        minimize(
            fun=fun,
            method=self.algorithm,
            jac=self.data_generator.dfdx,
            x0=self.data.get_n_best_output(1).to_numpy()[0].ravel(),
            callback=self._callback,
            options=self.hyperparameters,
            bounds=self.data.domain.get_bounds(),
            tol=0.0,
        )

    def _iterate(self, iterations: int, kwargs: dict,
                 x0_selection: str | ExperimentData,
                 sampler: Sampler, overwrite: bool,
                 callback: Callable, last_index: int):
        """Internal represenation of the iteration process for scipy-minimize
        optimizers.

        Parameters
        ----------
        optimizer : Optimizer
            Optimizer object
        data_generator : DataGenerator
            DataGenerator object
        iterations : int
            number of iterations
        kwargs : Dict[str, Any]
            any additional keyword arguments that will be passed to
            the DataGenerator
        x0_selection : str | ExperimentData
            How to select the initial design.
            The following x0_selections are available:

            * 'best': Select the best designs from the current experimentdata
            * 'random': Select random designs from the current experimentdata
            * 'last': Select the last designs from the current experimentdata
            * 'new': Create new random designs from the current experimentdata

            If the x0_selection is 'new', new designs are sampled with the
            sampler provided. The number of designs selected is equal to the
            population size of the optimizer.

            If an ExperimentData object is passed as x0_selection,
            the optimizer will use the input_data and output_data from this
            object as initial samples.

        sampler: Sampler
            If x0_selection = 'new', the sampler to use
        overwrite: bool
            If True, the optimizer will overwrite the current data.
        callback : Callable
            A callback function that is called after every iteration. It has
            the following signature:

                    ``callback(intermediate_result: ExperimentData)``

            where the first argument is a parameter containing an
            `ExperimentData` object with the current iterate(s).

        Raises
        ------
        ValueError
            Raised when invalid x0_selection is specified
        """
        n_data_before_iterate = deepcopy(len(self.data))
        if len(self.data) < self._population:
            raise ValueError(
                f'There are {len(self.data)} datapoints available, \
                        need {self._population} for initial \
                            population!'
            )

        self.run_algorithm(iterations)

        new_samples = self.data.select(self.data.index[1:])

        new_samples.evaluate(data_generator=self.data_generator,
                             mode='sequential', **kwargs)

        if callback is not None:
            callback(new_samples)

        if overwrite:
            self.data.add_experiments(
                self.data.select([self.data.index[-1]]))

        elif not overwrite:
            # Do not add the first element, as this is already
            # in the sampled data
            # self.data.add_experiments(new_samples)

            # TODO: At the end, the data should have
            # n_data_before_iterate + iterations amount of elements!
            # If x_new is empty, repeat best x0 to fill up total iteration
            if len(self.data) == n_data_before_iterate:
                repeated_sample = self.data.get_n_best_output(
                    n_samples=1)

                for repetition in range(iterations):
                    self.data.add_experiments(repeated_sample)

            # Repeat last iteration to fill up total iteration
            if len(self.data) < n_data_before_iterate + iterations:
                last_design = self.data.get_experiment_sample(len(self.data)-1)

                while len(self.data) < n_data_before_iterate + iterations:
                    self.data.add_experiments(last_design)

        self.data.evaluate(data_generator=self.data_generator,
                           mode='sequential', **kwargs)

        return self.data.select(self.data.index[self._population:])
