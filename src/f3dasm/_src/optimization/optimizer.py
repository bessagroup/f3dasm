"""
Module containing the interface class Optimizer
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from typing import Any, Callable, ClassVar, Dict, Literal, Optional

# Locals
from ..experimentdata.utils import number_of_overiterations, number_of_updates
from ._protocol import DataGenerator, ExperimentData, Sampler

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


# =============================================================================


class Optimizer:
    """
    Abstract class for optimization algorithms
    To implement a new optimizer, inherit from this class and implement the
    update_step method.

    Note
    ----
    The update_step method should have the following signature:

    '''
    def update_step(self)
    -> Tuple[np.ndarray, np.ndarray]:
    '''

    The method should return a tuple containing the new samples and the
    corresponding objective values. The new samples should be a numpy array.

    If the optimizer requires gradients, set the require_gradients attribute to
    True. This will ensure that the data_generator will calculate the gradients
    of the objective function from the DataGenerator.dfdx method.

    Hyperparameters can be set in the __init__ method of the child class.
    There are two hyperparameters that have special meaning:
    - population: the number of individuals in the population
    - seed: the seed of the random number generator

    You can create extra methods in your child class as you please, however
    it is advised not to create private methods (methods starting with an
    underscore) as these might be used in the base class.
    """
    type: ClassVar[str] = 'any'
    require_gradients: ClassVar[bool] = False

#                                                            Private Properties
# =============================================================================

    @property
    def _seed(self) -> int:
        """
        Property to return the seed of the optimizer

        Returns
        -------
        int | None
            Seed of the optimizer

        Note
        ----
        If the seed is not set, the property will return None
        This is done to prevent errors when the seed is not an available
        attribute in a custom optimizer class.
        """
        return self.seed if hasattr(self, 'seed') else None

    @property
    def _population(self) -> int:
        """
        Property to return the population size of the optimizer

        Returns
        -------
        int
            Number of individuals in the population

        Note
        ----
        If the population is not set, the property will return 1
        This is done to prevent errors when the population size is not an
        available attribute in a custom optimizer class.
        """
        return self.population if hasattr(self, 'population') else 1

#                                                                Public Methods
# =============================================================================

    def update_step(self) -> ExperimentData:
        """Update step of the optimizer. Needs to be implemented
         by the child class

        Returns
        -------
        ExperimentData
            ExperimentData object containing the new samples

        Raises
        ------
        NotImplementedError
            Raises when the method is not implemented by the child class

        Note
        ----
        You can access the data attribute of the optimizer to get the
        available data points. The data attribute is an
        f3dasm.ExperimentData object.
        """
        raise NotImplementedError(
            "You should implement an update step for your optimizer!")

    def init(self, data: ExperimentData, data_generator: DataGenerator):
        """
        Initialize the optimizer with the given data and data generator

        Parameters
        ----------
        data : ExperimentData
            Data to initialize the optimizer
        data_generator : DataGenerator
            Data generator to calculate the objective value
        """
        self.data = data
        self.data_generator = data_generator

    def call(self, iterations: int, last_index: int,
             kwargs: Optional[Dict[str, Any]] = None,
             x0_selection: Literal['best', 'random',
                                   'last',
                                   'new'] | ExperimentData = 'best',
             sampler: Optional[Sampler | str] = 'random',
             overwrite: bool = False,
             callback: Optional[Callable] = None,
             ) -> ExperimentData:

        return self._iterate(
            iterations=iterations, kwargs=kwargs,
            x0_selection=x0_selection,
            sampler=sampler,
            overwrite=overwrite,
            callback=callback,
            last_index=last_index)

    def _iterate(self, iterations: int, kwargs: Dict[str, Any],
                 x0_selection: str, sampler: Sampler, overwrite: bool,
                 callback: Callable, last_index: int):
        """Internal represenation of the iteration process

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
        if len(self.data) < self._population:
            raise ValueError(
                f'There are {len(self.data)} datapoints available, \
                        need {self._population} for initial \
                            population!'
            )

        for _ in range(number_of_updates(
                iterations,
                population=self._population)):
            new_samples = self.update_step()

            # If new_samples is a tuple of input_data and output_data
            if isinstance(new_samples, tuple):
                new_samples = type(self.data)(
                    domain=self.data.domain,
                    input_data=new_samples[0],
                    output_data=new_samples[1],
                )

            # If applicable, evaluate the new designs:
            new_samples.evaluate(data_generator=self.data_generator,
                                 mode='sequential', **kwargs)

            if callback is not None:
                callback(new_samples)

            if overwrite:
                _indices = new_samples.index + last_index + 1
                self.data._overwrite_experiments(experiment_sample=new_samples,
                                                 indices=_indices,
                                                 add_if_not_exist=True)

            else:
                self.data.add_experiments(new_samples)

        if not overwrite:
            # Remove overiterations
            self.data.remove_rows_bottom(number_of_overiterations(
                iterations,
                population=self._population))

        return self.data.select(self.data.index[self._population:])
