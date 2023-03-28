import fcntl
# from hydra.plugins.hydra_job_logging.handlers import Handler
from logging import FileHandler, StreamHandler
import time
import errno


class CustomFileHandler(FileHandler):
    def __init__(self, filename):
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

                lock_file(self.stream)
                StreamHandler.emit(self, record)

                unlock_file(self.stream)
                print("Succesfully logged!")

                break
            except IOError as e:
                # the file is locked by another process
                if e.errno == errno.EAGAIN:
                    print("The log file is currently locked by another process. Retrying in 1 second...")
                    time.sleep(1)
                else:
                    print(f"An unexpected IOError occurred: {e}")
                    break
            except Exception as e:
                # handle any other exceptions
                print(f"An unexpected error occurred: {e}")
                break


def lock_file(file):
    fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)


def unlock_file(file):
    fcntl.flock(file, fcntl.LOCK_UN)
