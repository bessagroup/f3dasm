"""Tests for custom error classes."""

from pathlib import Path

import pytest

from f3dasm._src.errors import (
    DecodeError,
    EmptyFileError,
    ReachMaximumTriesError,
    TimeOutError,
)

pytestmark = pytest.mark.smoke


class TestEmptyFileError:
    def test_basic(self):
        err = EmptyFileError("/tmp/test.csv")
        assert "File is empty" in str(err)
        assert "/tmp/test.csv" in str(err)
        assert err.file_path == Path("/tmp/test.csv")

    def test_custom_message(self):
        err = EmptyFileError("/tmp/test.csv", message="No data found")
        assert "No data found" in str(err)

    def test_is_exception(self):
        with pytest.raises(EmptyFileError):
            raise EmptyFileError("/tmp/test.csv")


class TestDecodeError:
    def test_basic(self):
        err = DecodeError("/tmp/test.json")
        assert "Error decoding file" in str(err)
        assert "/tmp/test.json" in str(err)

    def test_custom_message(self):
        err = DecodeError("/tmp/test.json", message="JSON parse failed")
        assert "JSON parse failed" in str(err)

    def test_default_path(self):
        err = DecodeError()
        assert "Error decoding file" in str(err)

    def test_is_exception(self):
        with pytest.raises(DecodeError):
            raise DecodeError("/tmp/x")


class TestReachMaximumTriesError:
    def test_basic(self):
        err = ReachMaximumTriesError("/tmp/data", max_tries=20)
        assert "Reached maximum number of tries" in str(err)
        assert "20" in str(err)
        assert err.max_tries == 20

    def test_custom_message(self):
        err = ReachMaximumTriesError(
            "/tmp/data", max_tries=5, message="Too many retries"
        )
        assert "Too many retries" in str(err)

    def test_is_exception(self):
        with pytest.raises(ReachMaximumTriesError):
            raise ReachMaximumTriesError("/tmp/x", 10)


class TestTimeOutError:
    def test_basic(self):
        err = TimeOutError(timeout=60)
        assert "Reached time-out" in str(err)
        assert "60 seconds" in str(err)
        assert err.timeout == 60

    def test_custom_message(self):
        err = TimeOutError(timeout=30, message="Operation timed out")
        assert "Operation timed out" in str(err)

    def test_is_exception(self):
        with pytest.raises(TimeOutError):
            raise TimeOutError(120)
