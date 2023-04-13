#                                                                       Modules
# =============================================================================

import errno
# Standard
import os
from logging import FileHandler, StreamHandler
from time import sleep
from typing import TextIO

if os.name == 'nt':
    import msvcrt
else:
    import fcntl

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class DistributedFileHandler(FileHandler):
    def __init__(self, filename):
        """Distributed FileHandler class for handling logging to
        one single file when multiple nodes access the same resource

        Parameters
        ----------
        filename
            name of the logging file
        """
        super().__init__(filename)

    def emit(self, record):
        """
        Emit a record.

        If the stream was not opened because 'delay' was specified in the
        constructor, open it before calling the superclass's emit.
        """
        while True:
            try:
                if self.stream is None:
                    self.stream = self._open()

                _lock_file(self.stream)
                StreamHandler.emit(self, record)

                _unlock_file(self.stream)
                print("Succesfully logged!")

                break
            except IOError as e:
                # the file is locked by another process
                if e.errno == errno.EAGAIN:
                    print("The log file is currently locked by another process. Retrying in 1 second...")
                    sleep(1)
                else:
                    print(f"An unexpected IOError occurred: {e}")
                    break
            except Exception as e:
                # handle any other exceptions
                print(f"An unexpected error occurred: {e}")
                break


def _lock_file(file):
    """Lock the file with the lock

    Parameters
    ----------
    file
        file object that returns from open()
    """
    if os.name == 'nt':  # for Windows
        msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)
    else:  # for Unix
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(file):
    """Unlock the file with the lock

    Parameters
    ----------
    file
        file object that returns from open()
    """
    if os.name == 'nt':  # for Windows
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:  # for Unix
        fcntl.flock(file, fcntl.LOCK_UN)
