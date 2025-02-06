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
