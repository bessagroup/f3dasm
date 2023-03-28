import sys
from unittest.mock import patch

import pytest

from f3dasm._show_versions import _get_deps_info

pytestmark = pytest.mark.smoke


def test_get_deps_info_with_existing_module():
    # Test that _get_deps_info() returns the correct version when the module exists

    with patch('importlib.import_module') as mock_import_module:
        # Configure the mock to return a mock object with a __version__ attribute
        mock_module = type('MockModule', (object,), {'__version__': '1.2.3'})
        mock_import_module.return_value = mock_module

        # Call the function and check that it returns the expected dictionary
        deps_info = _get_deps_info(['some_module'])
        assert deps_info == {'some_module': '1.2.3'}


def test_get_deps_info_with_no_version():
    # Test that _get_deps_info() returns 'No __version__ attribute!' when the module has no __version__ attribute

    with patch('importlib.import_module') as mock_import_module:
        # Configure the mock to return a mock object with no __version__ attribute
        mock_module = type('MockModule', (object,), {})
        mock_import_module.return_value = mock_module

        # Call the function and check that it returns the expected dictionary
        deps_info = _get_deps_info(['some_module'])
        assert deps_info == {'some_module': 'No __version__ attribute!'}


def test_get_deps_info_with_import_error():
    # Test that _get_deps_info() returns None when there is an ImportError

    with patch('importlib.import_module') as mock_import_module:
        # Configure the mock to raise an ImportError
        mock_import_module.side_effect = ImportError

        # Call the function and check that it returns None
        deps_info = _get_deps_info(['non_existing_module'])
        assert deps_info == {'non_existing_module': None}


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
