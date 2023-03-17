#                                                                       Modules
# =============================================================================

# Standard
import json

# Local
from ..optimization.all_optimizers import OPTIMIZERS
from ..optimization.optimizer import Optimizer
from ..design import create_experimentdata_from_json


#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def find_optimizer(query: str) -> Optimizer:
    """Find a optimizer from the f3dasm.optimizer submodule

    Parameters
    ----------
    query
        string representation of the requested optimizer

    Returns
    -------
        class of the requested optimizer
    """
    try:
        return list(filter(lambda optimizer: optimizer.__name__ == query, OPTIMIZERS))[0]
    except IndexError:
        return ValueError(f'Optimizer {query} not found!')


def create_optimizer_from_json(json_string: str):
    """Create an Optimizer object from a json string

    Parameters
    ----------
    json_string
        json string representation of the information to construct the Optimizer

    Returns
    -------
        Requested Optimizer object
    """
    optimizer_dict, name = json.loads(json_string)
    return create_optimizer_from_dict(optimizer_dict, name)


def create_optimizer_from_dict(optimizer_dict: dict, name: str) -> Optimizer:
    """Create an Optimizer object from a dictionary

    Parameters
    ----------
    optimizer_dict
        dictionary representation of the information to construct the Optimizer
    name
        name of the class

    Returns
    -------
        Requested Optimizer object
    """
    optimizer_dict['data'] = create_experimentdata_from_json(optimizer_dict['data'])
    return find_optimizer(name)(**optimizer_dict)
