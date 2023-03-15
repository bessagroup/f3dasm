from itertools import chain
from pathlib import Path

from setuptools import find_packages, setup

# # workround for enabling editable user pip installs
# site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

# version

here = Path(__file__).absolute().parent
version_data = {}
with open(here.joinpath("src", "f3dasm", "VERSION"), "r") as f:
    version = f.read()


# Get the long description from the README file
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# Get requirements


def read_requirements(file: str):

    with open(here.joinpath("requirements", f"{file}.txt"), "r", encoding="utf-8") as f:
        requirements = f.read().splitlines()

    return requirements


REQUIREMENTS_DEV = list(chain(read_requirements('documentation'),
                              read_requirements('test'), read_requirements('build')))

REQUIREMENTS_SAMPLING = read_requirements('sampling')
REQUIREMENTS_MACHINELEARNING = read_requirements('machinelearning')
REQUIREMENTS_OPTIMIZATION = read_requirements('optimization')

install_requires = read_requirements('core')
extra_requires = {"development": REQUIREMENTS_DEV,
                  "sampling": REQUIREMENTS_SAMPLING,
                  "machinelearning": REQUIREMENTS_MACHINELEARNING,
                  "optimization": REQUIREMENTS_OPTIMIZATION},

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
    classifiers=[

        "License :: OSI Approved :: BSD License",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    extras_require=extra_requires,
    keywords="recommendations recommendation recommenders recommender system engine "
    "machine learning python spark gpu",
    install_requires=install_requires,
    package_dir={'': "src"},
    python_requires=">=3.7, <3.11",
    # packages=find_packages(where="src", exclude=["contrib", "docs", "examples", "scenarios", "tests", "tools"]),
    packages=find_packages("src", exclude=["docs", "tests"]),
    # setup_requires=["numpy>=1.15"]
)
