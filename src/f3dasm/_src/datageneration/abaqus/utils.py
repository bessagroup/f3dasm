"""
Utility functions for abaqus datagenerator
"""

#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Alpha"
# =============================================================================
#
# =============================================================================


def remove_files(
    directory: str, file_types: list = [".log", ".lck", ".SMABulk", ".rec", ".SMAFocus",
                                        ".exception", ".simlog", ".023", ".exception"],
) -> None:
    """Remove files of specified types in a directory.

    Parameters
    ----------
    directory : str
        Target folder.
    file_types : list
        List of file extensions to be removed.
    """
    # Create a Path object for the directory
    dir_path = Path(directory)

    for target_file in file_types:
        # Use glob to find files matching the target extension
        target_files = dir_path.glob(f"*{target_file}")

        # Remove the target files if they exist
        for file in target_files:
            if file.is_file():
                file.unlink()
