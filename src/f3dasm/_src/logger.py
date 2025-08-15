"""
This module defines the logging settings for the f3dasm package.
"""
#                                                                       Modules
# =============================================================================

# Standard
import logging

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

# Create a logger for "f3dasm"
logger = logging.getLogger("f3dasm")

# Create a custom formatter for the "f3dasm" logger
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Create a custom handler for the "f3dasm" logger
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Set the default level for the "f3dasm" logger
logger.setLevel(logging.WARNING)

# Add the custom handler to the "f3dasm" logger
logger.addHandler(handler)
