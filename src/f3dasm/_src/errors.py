from __future__ import annotations

from pathlib import Path


class EmptyFileError(Exception):
    """Exception raised when a file exists but is empty."""

    def __init__(self, file_path: str | Path, message: str = "File is empty"):
        """
        Initializes the EmptyFileError.

        Args:
            file_path (str | Path): The path to the empty file.
            message (str): A custom error message.
        """
        self.file_path = Path(file_path)  # Ensure it's a Path object
        self.message = f"{message}: {self.file_path}"
        super().__init__(self.message)


class DecodeError(Exception):
    """Exception raised when opening a file gives errors"""

    def __init__(self, file_path: str | Path = '',
                 message: str = "Error decoding file"):
        """
        Initializes the EmptyFileError.

        Args:
            file_path (str | Path): The path to faulty file.
            message (str): A custom error message.
        """
        self.file_path = file_path  # Ensure it's a Path object
        self.message = f"{message}: {self.file_path}"
        super().__init__(self.message)


class ReachMaximumTriesError(Exception):
    """Exception raised when a function reaches its maximum number of tries."""

    def __init__(self, file_path: str | Path, max_tries: int,
                 message: str = "Reached maximum number of tries"):
        """
        Initializes the ReachMaximumTriesError.

        Args:
            max_tries (int): The maximum number of tries.
            message (str): A custom error message.
        """
        self.max_tries = max_tries
        self.message = f"{message} for {file_path}: {self.max_tries}"
        super().__init__(self.message)


class TimeOutError(Exception):
    """Exception raised when a function takes too long."""

    def __init__(self, timeout: int,
                 message: str = "Reached time-out"):
        """
        Initializes the TimeOutError.

        Args:
            max_tries (int): The maximum number of tries.
            message (str): A custom error message.
        """
        self.timeout = timeout
        self.message = (f"{message}: function timed-out after "
                        f"{self.timeout} seconds")
        super().__init__(self.message)
