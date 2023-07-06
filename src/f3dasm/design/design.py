#                                                                       Modules
# =============================================================================

# Standard
from typing import Any, Dict, Tuple

# Third-party
import numpy as np

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class Design:
    def __init__(self, dict_input: Dict[str, Any], dict_output: Dict[str, Any], jobnumber: int):
        self._dict_input = dict_input
        self._dict_output = dict_output
        self._jobnumber = jobnumber

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Converts the design to a tuple of numpy arrays.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple of numpy arrays containing the input and output data.
        """
        return np.array(list(self._dict_input.values())), np.array(list(self._dict_output.values()))

    @property
    def input_data(self) -> Dict[str, Any]:
        """Retrieve the input data of the design as a dictionary."""
        return self._dict_input

    @property
    def output_data(self) -> Dict[str, Any]:
        """Retrive the output data of the design as a dictionary."""
        return self._dict_output

    @property
    def job_number(self) -> int:
        """Retrieve the job number of the design."""
        return self._jobnumber

    def get(self, key: str) -> Any:
        # Check if key is in _dict_output but not in _dict_input
        if key in self._dict_output and key not in self._dict_input:
            # Raise keyerror
            raise KeyError(f"Variable '{key}' not found in input space. You can only access "
                           "variables that are in the input space.")

        return self._dict_input[key]

    def set(self, key: str, value: Any) -> None:
        # Check if key is in the _dict_input
        if key not in self._dict_output and key in self._dict_input:
            raise KeyError(f"Variable '{key}' not found in output space. You can only set "
                           "variables that are in the output space.")

        self._dict_output[key] = value
