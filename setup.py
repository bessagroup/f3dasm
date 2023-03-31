from itertools import chain
from pathlib import Path
from typing import List

from setuptools import find_packages, setup

here = Path(__file__).absolute().parent

# version
# version = here.joinpath("src", "f3dasm", "VERSION").read_text().strip()
version = '0.9.2'

# Get the long description from the README file
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# Get requirements


def read_requirements(file: Path) -> List[str]:
    with open(here.joinpath(file), "r", encoding="utf-8") as f:
        requirements = f.read().splitlines()
    return requirements


REQUIREMENTS_SAMPLING = read_requirements(Path('requirements', 'sampling.txt'))
REQUIREMENTS_MACHINELEARNING = read_requirements(Path('requirements', 'machinelearning.txt'))
REQUIREMENTS_OPTIMIZATION = read_requirements(Path('requirements', 'optimization.txt'))

install_requires = read_requirements(Path('requirements.txt'))
extras_require = {"sampling": REQUIREMENTS_SAMPLING,
                  "machinelearning": REQUIREMENTS_MACHINELEARNING,
                  "optimization": REQUIREMENTS_OPTIMIZATION,
                  }

# for the brave of heart
extras_require["all"] = list(set(sum([*extras_require.values()], [])))

# for the developers
extras_require["dev"] = list(chain(extras_require["all"],
                                   read_requirements(Path('docs', 'requirements.txt')),
                                   read_requirements(Path('tests', 'requirements.txt')),
                                   ["flake8"]))
setup(
    name="f3dasm",
    version=version,
    description="f3dasm - Framework for Data-driven development and Analysis of Structures and Materials",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/bessagroup/F3DASM",
    project_urls={
        "Documentation": "https://bessagroup.github.io/F3DASM/",
        "Wiki": "https://github.com/bessagroup/F3DASM/wiki",
    },
    author="Martin van der Schelling",
    author_email="M.P.vanderSchelling@tudelft.nl",
    license="BSD",
    classifiers=[

        "License :: OSI Approved :: BSD License",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    extras_require=extras_require,
    keywords="data-driven materials framework, machine learning",
    install_requires=install_requires,
    package_dir={'': "src"},
    python_requires=">=3.8, <3.11",
    packages=find_packages("src", exclude=["docs", "tests"]),
)
