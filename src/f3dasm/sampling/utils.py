#                                                                       Modules
# =============================================================================

# Standard
import json
from typing import List

# Locals
from ..design import create_design_from_json
from .all_samplers import SAMPLERS
from .sampler import Sampler

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def find_sampler(query: str) -> Sampler:
    """Find a Sampler from the f3dasm.design submodule

    Parameters
    ----------
    query
        string representation of the requested sampler

    Returns
    -------
        class of the requested sampler
    """
    try:
        return list(filter(lambda parameter: parameter.__name__ == query, SAMPLERS))[0]
    except IndexError:
        return ValueError(f'Sampler {query} not found!')


def create_sampler_from_json(json_string: str) -> Sampler:
    """Create a Sampler object from a json string

    Parameters
    ----------
    json_string
        json string representation of the information to construct the Parameter

    Returns
    -------
        Requested Sampler object
    """
    sampler_dict, name = json.loads(json_string)
    return _create_sampler_from_dict(sampler_dict, name)


def _create_sampler_from_dict(sampler_dict: dict, name: str) -> Sampler:
    sampler_dict['design'] = create_design_from_json(sampler_dict['design'])
    return find_sampler(name)(**sampler_dict)
