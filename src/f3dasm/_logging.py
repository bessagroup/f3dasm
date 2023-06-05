"""
This module defines the logging settings for the f3dasm package.
"""
#                                                                       Modules
# =============================================================================

# Standard
import logging

# Local
from ._show_versions import __version__

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

# Logging things
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def create_logger(name: str, level: int = logging.INFO, filename: str = None) -> logging.Logger:
    """Create a logger

    Parameters
    ----------
    name : str
        Name of the logger
    level : int, optional
        Logging level, by default logging.INFO
    filename : str, optional
        Filename of the log file, by default None

    Returns
    -------
    logging.Logger
        The created logger
    """

    # Create a logger
    logger = logging.getLogger(name)

    # Set the logging level
    logger.setLevel(level)

    # Create a file handler
    if filename is None:
        filename = f"{name}.log"

    # Check if file ends with .log
    if not filename.endswith(".log"):
        filename += ".log"

    handler = logging.FileHandler(filename)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)

    return logger


# Create a logger
logger = create_logger('f3dasm', level=logging.INFO)

# Log welcome message and the version of f3dasm
logger.info(f"Imported f3dasm (version: {__version__})")
