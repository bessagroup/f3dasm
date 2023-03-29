#                                                                       Modules
# =============================================================================

# Standard
import fcntl
from logging import FileHandler, StreamHandler
from time import sleep
import errno
from typing import TextIO

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


def _lock_file(file: TextIO):
    """Lock the file with the fcntl lock

    Parameters
    ----------
    file
        file object that returns from open()
    """
    fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(file: TextIO):
    """Unlock the file with the fcntl lock

    Parameters
    ----------
    file
        file object that returns from open()
    """
    fcntl.flock(file, fcntl.LOCK_UN)
