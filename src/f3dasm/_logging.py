"""
This module defines the logging settings for the f3dasm package.
"""
#                                                                       Modules
# =============================================================================

# Standard
import logging

# Local
from ._show_versions import show_versions

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

logging.info("Imported f3dasm")
logging.info(show_versions())
