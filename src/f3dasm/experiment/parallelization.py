#                                                                       Modules
# =============================================================================


# Standard
from typing import Any, Callable, Dict, Iterator, List, Protocol, Tuple

# Third-party core
from pathos.helpers import mp

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class ExperimentData(Protocol):
    def __iter__(self) -> Iterator[Tuple[Dict[str, Any]]]:
        ...


def run_operation_on_experiments(data: ExperimentData, operation: Callable,
                                 parallel: bool = False, **kwargs) -> List[Any]:
    """Run an operation on a list of experiments

    Parameters
    ----------
    data
        ExperimtentData object
    operation
        Callable function that accepts one ExperimentData line
    parallel, optional
        Flag if the operation should be done in parallel, by default False

    Returns
    -------
        List of returns from the operation
    """

    if parallel:
        options = [
            ({'index': index, 'value_input': value_input, 'value_output': value_output, **kwargs},)
            for index, (value_input, value_output) in enumerate(data)
        ]

        def f(options: Dict[str, Any]) -> Any:
            """This function wraps the operation to unpack the options dictionary"""
            return operation(**options)

        with mp.Pool() as pool:
            # maybe implement pool.starmap_async ?
            results = pool.starmap(f, options)

    else:
        results = []
        for index, (value_input, value_output) in enumerate(data):
            results.append(operation(index, value_input, value_output, **kwargs))

    return results
