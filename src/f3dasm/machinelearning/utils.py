#                                                                       Modules
# =============================================================================

# Standard
import json

# Local
from .all_models import MODELS
from .model import Model

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def find_model(query: str) -> Model:
    """Find a machine learning model from the f3dasm.machinelearning submodule

    Parameters
    ----------
    query
        string representation of the requested model

    Returns
    -------
        class of the requested model
    """
    try:
        return list(filter(lambda model: model.__name__ == query, MODELS))[0]
    except IndexError:
        return ValueError(f'Model {query} not found!')


def create_model_from_json(json_string: str):
    """Create a Model object from a json string

    Parameters
    ----------
    json_string
        json string representation of the information to construct the Model

    Returns
    -------
        Requested Model object
    """
    function_dict, name = json.loads(json_string)
    return create_model_from_dict(function_dict, name)


def create_model_from_dict(model_dict: dict, name: str) -> Model:
    """Create a Model object from a dictionary

    Parameters
    ----------
    model_dict
        dictionary representation of the information to construct the Model
    name
        name of the class

    Returns
    -------
        Requested Model object
    """
    return find_model(name)(**model_dict)
