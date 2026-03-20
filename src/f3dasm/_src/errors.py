from __future__ import annotations

from pathlib import Path


class EmptyFileError(Exception):
    """Exception raised when a file exists but is empty."""

    def __init__(self, file_path: str | Path, message: str = "File is empty"):
        """
        Initializes the EmptyFileError.

        Parameters
        ----------
        file_path : str or Path
            The path to the empty file.
        message : str, optional
            A custom error message, by default "File is empty".
        """
        self.file_path = Path(file_path)  # Ensure it's a Path object
        self.message = f"{message}: {self.file_path}"
        super().__init__(self.message)


class DecodeError(Exception):
    """Exception raised when opening a file gives errors"""

    def __init__(
        self, file_path: str | Path = "", message: str = "Error decoding file"
    ):
        """
        Initializes the DecodeError.

        Parameters
        ----------
        file_path : str or Path, optional
            The path to faulty file, by default "".
        message : str, optional
            A custom error message, by default "Error decoding file".
        """
        self.file_path = file_path  # Ensure it's a Path object
        self.message = f"{message}: {self.file_path}"
        super().__init__(self.message)


class ReachMaximumTriesError(Exception):
    """Exception raised when a function reaches its maximum number of tries."""

    def __init__(
        self,
        file_path: str | Path,
        max_tries: int,
        message: str = "Reached maximum number of tries",
    ):
        """
        Initializes the ReachMaximumTriesError.

        Parameters
        ----------
        file_path : str or Path
            The path to the file that was being accessed.
        max_tries : int
            The maximum number of tries.
        message : str, optional
            A custom error message,
            by default "Reached maximum number of tries".
        """
        self.max_tries = max_tries
        self.message = f"{message} for {file_path}: {self.max_tries}"
        super().__init__(self.message)


class TimeOutError(Exception):
    """Exception raised when a function takes too long."""

    def __init__(self, timeout: int, message: str = "Reached time-out"):
        """
        Initializes the TimeOutError.

        Parameters
        ----------
        timeout : int
            The timeout duration in seconds.
        message : str, optional
            A custom error message, by default "Reached time-out".
        """
        self.timeout = timeout
        self.message = (
            f"{message}: function timed-out after {self.timeout} seconds"
        )
        super().__init__(self.message)
