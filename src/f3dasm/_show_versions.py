"""
Utility methods to print system info for debugging

adapted from :func:`sklearn.show_versions`
"""
# License: BSD 3 clause

import importlib
import platform
import sys
from typing import Any, List


def _get_sys_info() -> dict:
    """System information

    Returns
    -------
    sys_info : dict
        system and Python version information

    """
    python = sys.version.replace("\n", " ")

    blob = [
        ("python", python),
        ("executable", sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)


def _get_deps_info(deps: List[str]) -> dict:
    """Overview of the installed version of main dependencies

    Parameters
    ----------
    deps: List[str]
        list of dependencies

    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries

    """

    def get_version(module) -> str:
        try:
            return module.__version__
        except AttributeError:
            return 'No __version__ attribute!'

    deps_info = {}

    for modname in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
            ver = get_version(mod)
            deps_info[modname] = ver
        except ImportError:
            deps_info[modname] = None

    return deps_info


def show_versions():
    """Print useful debugging information"

    .. versionadded:: 0.20
    """

    deps = [
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'SALib',
        'hypothesis',
        'GPyOpt',
        'GPy',
        'autograd',
        'numdifftools',
        'tensorflow',
        'pathos',
        'pytest',
        'hypothesis',
        'pygmo',
    ]

    sys_info = _get_sys_info()
    f3dasm_info = _get_deps_info(['f3dasm'])
    deps_info = _get_deps_info(deps)

    print("\nF3DASM:")
    for k, stat in f3dasm_info.items():
        print(f"{k:>13}: {stat}")

    print("\nSystem:")
    for k, stat in sys_info.items():
        print(f"{k:>10}: {stat}")

    print("\nPython dependencies:")
    for k, stat in deps_info.items():
        print(f"{k:>13}: {stat}")
