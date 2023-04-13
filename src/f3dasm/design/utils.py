#                                                                       Modules
# =============================================================================

# Standard
import json

# Third-party core
import pandas as pd

# Local
from .all_parameters import PARAMETERS
from .design import DesignSpace, make_nd_continuous_design
from .experimentdata import ExperimentData
from .parameter import Parameter

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class F3DASMClassNotFoundError(Exception):
    """
    Exception raised when a class is not found.

    Attributes:
        message (str): The error message.
    """

    def __init__(self, message):
        super().__init__(message)


def find_parameter(query: str) -> Parameter:
    """
    Find a Parameter class by its name.

    Parameters
    ----------
    query : str
        The name of the requested Parameter class.

    Returns
    -------
    Parameter
        The class object of the requested Parameter.

    Raises
    ------
    F3DASMClassNotFoundError
        If the requested parameter is not found.
    """
    try:
        return list(filter(lambda parameter: parameter.__name__ == query, PARAMETERS))[0]
    except IndexError:
        return F3DASMClassNotFoundError(f'Parameter {query} not found!')


def create_parameter_from_json(json_string: str):
    """
    Create a Parameter object from a JSON string.

    Parameters
    ----------
    json_string : str
        JSON string representation of the information to construct the Parameter.

    Returns
    -------
    Parameter
        The requested Parameter object.
    """
    parameter_dict, name = json.loads(json_string)
    return _create_parameter_from_dict(parameter_dict, name)


def _create_parameter_from_dict(parameter_dict: dict, name: str) -> Parameter:
    """
    Create a Parameter object from a dictionary.

    Parameters
    ----------
    parameter_dict : dict
        Dictionary representation of the information to construct the Parameter.
    name : str
        Name of the class.

    Returns
    -------
    Parameter
        The requested Parameter object.
    """
    return find_parameter(name)(**parameter_dict)


# Create designspace from json file
def create_design_from_json(json_string: str) -> DesignSpace:
    """
    Create a DesignSpace object from a JSON string.

    Parameters
    ----------
    json_string : str
        JSON string encoding the DesignSpace object.

    Returns
    -------
    DesignSpace
        The created DesignSpace object.
    """
    # Load JSON string
    design_dict = json.loads(json_string)
    return _create_design_from_dict(design_dict)


def _create_design_from_dict(design_dict: dict) -> DesignSpace:
    """
    Create a DesignSpace object from a dictionary.

    Parameters
    ----------
    design_dict : dict
        Dictionary representation of the information to construct the DesignSpace.

    Returns
    -------
    DesignSpace
        The created DesignSpace object.
    """
    for key, space in design_dict.items():
        parameters = []
        for parameter in space:
            parameters.append(create_parameter_from_json(parameter))

        design_dict[key] = parameters

    return DesignSpace(**design_dict)


def create_experimentdata_from_json(json_string: str) -> ExperimentData:
    """
    Create an ExperimentData object from a JSON string.

    Parameters
    ----------
    json_string : str
        JSON string encoding the ExperimentData object.

    Returns
    -------
    ExperimentData
        The created ExperimentData object.
    """
    # Read JSON
    experimentdata_dict = json.loads(json_string)
    return _create_experimentdata_from_dict(experimentdata_dict)


def _create_experimentdata_from_dict(experimentdata_dict: dict) -> ExperimentData:
    """
    Create an ExperimentData object from a dictionary.

    Parameters
    ----------
    experimentdata_dict : dict
        Dictionary representation of the information to construct the ExperimentData.

    Returns
    -------
    ExperimentData
        The created ExperimentData object.
    """
    # Read design from json_data_loaded
    new_design = create_design_from_json(experimentdata_dict['design'])

    # Read data from json string
    new_data = pd.read_json(experimentdata_dict['data'])

    # Create tuples of indices
    columntuples = tuple(tuple(entry[1:-1].replace("'", "").split(', ')) for entry in new_data.columns.values)

    # Create MultiIndex object
    columnlabels = pd.MultiIndex.from_tuples(columntuples)

    # Overwrite columnlabels
    new_data.columns = columnlabels

    # Create
    new_experimentdata = ExperimentData(design=new_design)
    new_experimentdata.add(data=new_data)

    return new_experimentdata


def load_experimentdata(filename: str) -> ExperimentData:
    """Load an ExperimentData object from .csv and .json files.

    Parameters
    ----------
    filename : str
        Name of the file, excluding suffix.

    Returns
    -------
    ExperimentData
        ExperimentData object containing the loaded data.
    """
    # Load the design from a json string
    with open(f"{filename}_design.json") as f:
        design = create_design_from_json(f.read())

    # Load the data from a csv
    data = pd.read_csv(f"{filename}_data.csv", header=[0, 1], index_col=0)

    # Create the experimentdata object
    experimentdata = ExperimentData(design=design)
    experimentdata.data = data
    return experimentdata
