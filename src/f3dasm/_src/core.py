"""
This module contains the core blocks and protocols for the f3dasm package.
"""
#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, Optional

# Third-party
from hydra.utils import instantiate
from omegaconf import DictConfig

from .datagenerator import (
    evaluate_cluster,
    evaluate_cluster_array,
    evaluate_mpi,
    evaluate_multiprocessing,
    evaluate_sequential,
)
from .design.domain import Domain

# Local
from .experimentdata import ExperimentData
from .experimentsample import ExperimentSample

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Alpha"
# =============================================================================
#
# =============================================================================


class Block(ABC):
    """Abstract base class representing a single operation in a data-driven
    pipeline.

    A Block transforms an :class:`ExperimentData` object and returns the
    updated result. Subclasses must implement the :meth:`call` method.
    Optionally, they may override :meth:`arm` to perform any one-time setup
    (e.g. fitting a surrogate model) before the block is executed.

    Blocks compose with the ``>>`` operator into a :class:`ChainedBlock`,
    and can be repeated with :meth:`loop` to produce a :class:`LoopBlock`.
    The built-in samplers (:class:`RandomUniform`, :class:`Latin`,
    :class:`Sobol`, :class:`Grid`), the :class:`DataGenerator`, and the
    built-in optimizer update steps are all Block subclasses.

    Examples
    --------
    Define a custom block:

    >>> class MyBlock(Block):
    ...     def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
    ...         # transform data here
    ...         return data
    """

    def arm(self, data: ExperimentData) -> None:
        """
        Prepare the block with a given ExperimentData.

        Parameters
        ----------
        data : ExperimentData
            The experiment data to be used by the block.

        Notes
        -----
        This method can be inherited by a subclasses to prepare the block
        with the given experiment data. It is not required to implement this
        method in the subclass.
        """
        pass

    @abstractmethod
    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        """
        Execute the block's operation on the ExperimentData.

        Parameters
        ----------
        data : ExperimentData
            The experiment data to process.
        **kwargs : dict
            Additional keyword arguments for the operation.

        Returns
        -------
        ExperimentData
            The processed experiment data.
        """
        pass

    def __rshift__(self, other: Block) -> ChainedBlock:
        """Chain two blocks with the ``>>`` operator.

        Parameters
        ----------
        other : Block
            The block to execute after this one.

        Returns
        -------
        ChainedBlock
            A new block that runs ``self`` then ``other``.
        """
        if isinstance(other, ChainedBlock):
            return ChainedBlock(blocks=[self, *other.blocks])
        return ChainedBlock(blocks=[self, other])

    def loop(self, n_iterations: int) -> LoopBlock:
        """Repeat this block ``n_iterations`` times.

        Parameters
        ----------
        n_iterations : int
            Number of times to run the block.

        Returns
        -------
        LoopBlock
            A new block that runs ``self`` ``n_iterations`` times, passing
            the output of each iteration as the input to the next.

        Examples
        --------
        >>> step = update_step >> data_generator
        >>> data = step.loop(50).call(initial_data)
        """
        return LoopBlock(block=self, n_iterations=n_iterations)

    @classmethod
    def from_yaml(
        cls, init_config: DictConfig, call_config: Optional[DictConfig] = None
    ) -> Block:
        """
        Create a block from a YAML configuration.

        Parameters
        ----------
        init_config : DictConfig
            The configuration for the block's initialization.
        call_config : DictConfig, optional
            The configuration for the block's call method, by default None

        Returns
        -------
        Block
            The block object created from the configuration.
        """
        block: Block = instantiate(init_config)
        if call_config is not None:
            block.call = partial(block.call, **call_config)

        return block


class ChainedBlock(Block):
    """A block that runs multiple blocks in sequence.

    Created automatically by the ``>>`` operator on
    :class:`Block` instances.

    Parameters
    ----------
    blocks : list[Block]
        The blocks to execute in order.
    """

    def __init__(self, blocks: list[Block]) -> None:
        """Initialize the chained block.

        Parameters
        ----------
        blocks : list[Block]
            Ordered list of blocks to execute sequentially.
        """
        self.blocks = blocks

    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        """Execute all chained blocks in order.

        Parameters
        ----------
        data : ExperimentData
            The experiment data to process.
        **kwargs : dict
            Additional keyword arguments for each block.

        Returns
        -------
        ExperimentData
            The processed experiment data.
        """
        for block in self.blocks:
            data = block.call(data=data, **kwargs)
        return data

    def arm(self, data: ExperimentData) -> None:
        """Arm all chained blocks."""
        for block in self.blocks:
            block.arm(data)

    def __rshift__(self, other: Block) -> ChainedBlock:
        if isinstance(other, ChainedBlock):
            return ChainedBlock(blocks=[*self.blocks, *other.blocks])
        return ChainedBlock(blocks=[*self.blocks, other])


class LoopBlock(Block):
    """A block that repeats an inner block ``n_iterations`` times.

    Each iteration passes the output of the previous call as the input to
    the next. Typical use is to drive an optimizer update step chained with
    a data generator:

    >>> loop = (update_step >> data_generator).loop(50)
    >>> data = loop.call(initial_data)

    Parameters
    ----------
    block : Block
        The inner block to repeat.
    n_iterations : int
        Number of iterations to run.
    """

    def __init__(self, block: Block, n_iterations: int) -> None:
        self.block = block
        self.n_iterations = n_iterations

    def arm(self, data: ExperimentData) -> None:
        """Arm the inner block once before the loop starts."""
        self.block.arm(data)

    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        """Run the inner block ``n_iterations`` times sequentially.

        Parameters
        ----------
        data : ExperimentData
            The experiment data to process.
        **kwargs : dict
            Additional keyword arguments forwarded to each iteration.

        Returns
        -------
        ExperimentData
            The experiment data after ``n_iterations`` iterations.
        """
        for _ in range(self.n_iterations):
            data = self.block.call(data=data, **kwargs)
        return data


# =============================================================================


class DataGenerator:
    """Base class for generating output data for experiment samples.

    Subclasses must implement the :meth:`execute` method, which receives a
    single :class:`ExperimentSample` and returns it after storing the computed
    outputs.  The :meth:`call` method drives execution across a full
    :class:`ExperimentData` object and supports several parallelisation
    backends (sequential, multiprocessing, cluster file-lock, MPI).

    Examples
    --------
    Define a custom data generator:

    >>> class MyDataGenerator(DataGenerator):
    ...     def execute(
    ...         self, experiment_sample: ExperimentSample, **kwargs
    ...     ) -> ExperimentSample:
    ...         x0 = experiment_sample.input_data['x0']
    ...         experiment_sample.store('y', x0 ** 2)
    ...         return experiment_sample
    """

    def arm(self, data: ExperimentData) -> None:
        """Prepare the data generator before execution.

        Parameters
        ----------
        data : ExperimentData
            The experiment data that the generator will operate on.

        Notes
        -----
        Override this method in a subclass to perform one-time setup that
        requires access to the full :class:`ExperimentData` (e.g. fitting a
        surrogate model). The default implementation does nothing.
        """
        pass

    # =========================================================================

    @abstractmethod
    def execute(
        self, experiment_sample: ExperimentSample, **kwargs
    ) -> ExperimentSample:
        """Interface function that handles the execution of the data generator

        Parameters
        ----------
        experiment_sample : ExperimentSample
            The experiment sample to run the data generator on
        kwargs : dict
            The optional keyword arguments to pass to the function

        Returns
        -------
        ExperimentSample
            The experiment sample with the response of the data generator
            saved in the experiment_sample

        Raises
        ------
        NotImplementedError
            If the function is not implemented by the user
        """
        raise NotImplementedError(
            "The execute function of the DataGenerator must be "
            "implemented by the user."
        )

    def call(
        self,
        data: ExperimentData,
        mode: str = "sequential",
        pass_id: bool = False,
        **kwargs,
    ) -> ExperimentData:
        """
        Evaluate the data generator.

        Parameters
        ----------
        data : ExperimentData
            The experiment data to process.
        mode : str, optional
            The mode of evaluation, by default 'sequential'
        pass_id : bool, optional
            Whether to pass the id to the execute function, by default False
        **kwargs : dict
            The keyword arguments to pass to execute function

        Returns
        -------
        ExperimentData
            The processed data

        Raises
        ------
        ValueError
            If an invalid mode is specified

        Notes
        -----
        The mode can be one of the following:
            - 'sequential': Run the data generator sequentially
            - 'parallel': Run the data generator in parallel
            - 'cluster': Run the data generator on a cluster
            - 'mpi': Run the data generator using MPI
            - 'cluster_array': Run the data generator on a cluster array

        The 'pass_id' parameter is used to pass the id of the experiment sample
        to the execute function. This is useful when the execute function
        requires the id of the experiment sample to run. By default, this is
        set to False. The id is passed through the 'id' keyword argument.
        """
        data = data._copy(in_place=False, deep=True)

        if mode == "sequential":
            return evaluate_sequential(
                execute_fn=self.execute, data=data, pass_id=pass_id, **kwargs
            )
        elif mode == "parallel":
            return evaluate_multiprocessing(
                execute_fn=self.execute, data=data, pass_id=pass_id, **kwargs
            )
        elif mode.lower() == "cluster":
            return evaluate_cluster(
                execute_fn=self.execute, data=data, pass_id=pass_id, **kwargs
            )
        elif mode.lower() == "mpi":
            return evaluate_mpi(
                execute_fn=self.execute, data=data, pass_id=pass_id, **kwargs
            )
        elif mode.lower() == "cluster_array":
            return evaluate_cluster_array(
                execute_fn=self.execute, data=data, pass_id=pass_id, **kwargs
            )
        else:
            raise ValueError(f"Invalid parallelization mode specified: {mode}")


def datagenerator(
    output_names: Iterable[str], domain: Domain | None = None
) -> Callable[[Callable[..., Any]], DataGenerator]:
    """
    Decorator to convert a function into a `DataGenerator` subclass.

    The decorated function should take named arguments matching the keys in
    the Domain and return one or multiple output values. These values will be
    stored in the `ExperimentData` under the names specified in `output_names`.

    Parameters
    ----------
    output_names : Iterable[str]
        A list of names for the returned values of the decorated function.
        The function's return values will be stored in the `ExperimentSample`
        object under these names.
    domain : Domain, optional
        The domain describing the input and output space. If provided, it can
        be used for saving the data to disk or other domain-specific
        operations.

    Returns
    -------
    Callable[[Callable[..., Any]], DataGenerator]
        A decorator that transforms a function into a `DataGenerator` subclass.

    Raises
    ------
    ValueError
        If `output_names` is not provided.

    Examples
    --------
    >>> @datagenerator(output_names=['y'])
    ... def my_function(x0: float, x1: float, x2: float) -> float:
    ...     return x0**2 + x1**2 + x2**2
    ...
    >>> experiment_data = ExperimentData(domaind=domain)
    >>> experiment_data = my_function.call(experiment_data)
    """

    if not output_names:
        raise ValueError(
            "If you provide a function as a data generator, you must "
            "provide the names of the return arguments with the `output_names`"
            "attribute."
        )

    if domain is None:
        domain = Domain()

    # If the output names is a single string, convert it to a list
    if isinstance(output_names, str):
        output_names = [output_names]

    def decorator(f: Callable[..., Any]) -> DataGenerator:
        signature = inspect.signature(f)

        class FunctionDataGenerator(DataGenerator):
            """
            Auto-generated DataGenerator subclass from a decorated function.
            """

            def execute(
                self, experiment_sample: ExperimentSample, **kwargs: Any
            ) -> ExperimentSample:
                """
                Executes the data generation process by calling the decorated
                function.

                Parameters
                ----------
                experiment_sample : ExperimentSample
                    The experiment sample containing input data.
                **kwargs : dict
                    Additional keyword arguments to override values in
                    `experiment_sample.input_data`.

                Returns
                -------
                ExperimentSample
                    The modified `ExperimentSample` instance with new stored
                    outputs.
                """
                _input = {
                    name: val.default
                    for name, val in signature.parameters.items()
                    if val.default is not inspect.Parameter.empty
                }

                _input.update(kwargs)

                _input.update(
                    {
                        name: experiment_sample.input_data[name]
                        for name in experiment_sample.input_data
                        if name in signature.parameters
                    }
                )

                # Call the function
                _output = f(**_input)

                # Ensure the output is iterable
                if not isinstance(_output, tuple):
                    _output = (_output,)

                # Store outputs in the experiment sample
                for name, value in zip(output_names, _output, strict=False):
                    if name in domain.output_names:
                        parameter = domain.output_space[name]
                        experiment_sample.store(
                            name=name,
                            object=value,
                            to_disk=parameter.to_disk,
                            store_function=parameter.store_function,
                            load_function=parameter.load_function,
                            load_kwargs=parameter.load_kwargs,
                        )
                    else:
                        experiment_sample.store(name=name, object=value)

                return experiment_sample

        data_generator = FunctionDataGenerator()
        data_generator.output_names = output_names
        data_generator.f = f
        return data_generator

    return decorator
