"""
Utility methods to print system info for debugging

adapted from :func:`sklearn.show_versions`
"""
#                                                                       Modules
# =============================================================================

# Standard
import importlib
import platform
import sys
from pathlib import Path
from typing import List

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling', 'Scikit-learn']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

# version
here = Path(__file__).absolute().parent
with open(here.joinpath("VERSION"), "r") as f:
    __version__ = f.read()

# List of the dependencies per extension:
CORE_DEPS = [
    'numpy',
    'scipy',
    'pandas',
    'matplotlib',
    'pathos',
    'hydra',
    'autograd',
]

OPTIMIZATION_DEPS = [
    'GPyOpt',
    'GPy',
    'tensorflow',
    'pygmo',
]

MACHINELEARNING_DEPS = [
    'tensorflow',
]

SAMPLING_DEPS = [
    'SALib',
]


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
    """

    sys_info = _get_sys_info()
    f3dasm_info = _get_deps_info(['f3dasm'])
    core_deps_info = _get_deps_info(CORE_DEPS)
    optimization_deps_info = _get_deps_info(OPTIMIZATION_DEPS)
    machinelearning_deps_info = _get_deps_info(MACHINELEARNING_DEPS)
    sampling_deps_info = _get_deps_info(SAMPLING_DEPS)

    print("\nf3dasm:")
    for k, stat in f3dasm_info.items():
        print(f"{k:>13}: {stat}")

    print("\nSystem:")
    for k, stat in sys_info.items():
        print(f"{k:>10}: {stat}")

    print("\nCore package dependencies:")
    for k, stat in core_deps_info.items():
        print(f"{k:>13}: {stat}")

    print("\nMachine learning extension:")
    for k, stat in machinelearning_deps_info.items():
        print(f"{k:>13}: {stat}")

    print("\nOptimization extension:")
    for k, stat in optimization_deps_info.items():
        print(f"{k:>13}: {stat}")

    print("\nSampling extension:")
    for k, stat in sampling_deps_info.items():
        print(f"{k:>13}: {stat}")
