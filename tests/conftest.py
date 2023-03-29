# Source for the markers:
# https://doc.pytest.org/en/latest/example/markers.html#custom-marker-and-command-line-option-to-control-test-runs

import logging

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "-S",
        action="store",
        metavar="NAME",
        help="exclude tests with the dependency NAME.",
    )


def pytest_runtest_setup(item):
    """Pytest setup"""
    dependency_names = [mark.args[0] for mark in item.iter_markers(name="requires_dependency")]
    if not dependency_names:
        return

    if item.config.getoption("-S") in dependency_names or item.config.getoption("-S") == "all":
        pytest.skip(f"test skipped: requires dependency {dependency_names!r}")


@pytest.fixture(scope='session', autouse=True)
def setup_logging():
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
