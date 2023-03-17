#                                                                       Modules
# =============================================================================

# Standard
import json

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def find_class(module, query: str):
    """Find a class from a string

    Parameters
    ----------
    module
        (sub)module to be searching
    query
        string to search for

    Returns
    -------
        class
    """
    return getattr(module, query)


def write_json(name: str, json_string: str):
    """Write a JSON-strint to a file

    Parameters
    ----------
    name
        name of file toe write without file extension .json
    json_string
        JSON string to store
    """

    with open(f"{name}.json", "w", encoding='utf-8') as f:
        json.dump(json_string, f, ensure_ascii=False)
