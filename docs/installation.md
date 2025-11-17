# Installation

There are different ways to install `f3dasm`:

- [Install the latest official release](#install-the-latest-release). This is the best approach for most users who just want to use the package.  
- [Build the package from source](#install-from-source). This is for users who wish to contribute to the project.

---

## Install the latest release

`f3dasm` is purely Python code and compatible with:

1. Python 3.10 or higher  
2. Windows, macOS, and Linux  
3. The `pip` package manager

To install the latest release via pip:

```bash
pip install -U f3dasm
```

f3dasm is also available on [conda-forge](https://anaconda.org/conda-forge/f3dasm).

```bash
conda install conda-forge::f3dasm
```

## Install from source

The Python PyPI package (pip install f3dasm) contains the code that is used when installing the package as a *user*. It contains only the main branch version. Installing the package from source is mainly for *developers* and besides the source code it includes:

1. Studies
2. Test suite
3. Documentation source

Building from source is required to work on a contribution (bug fix, new feature, code or documentation improvement). We recommend using a Linux distribution system.

1. Use Git to check out the latest source from the f3dasm repository on Github:

```bash
git clone https://github.com/bessagroup/f3dasm.git  # add --depth 1 if your connection is slow
cd f3dasm
```

2. Install a recent version of Python (3.10 or higher). If you installed Python with conda, we recommend to create a dedicated conda environment with all the build dependencies of f3dasm:

```bash
conda create -n f3dasm_env python=3.10
conda activate f3dasm_env
```

3. If you run the development version, it is annoying to reinstall the package each time you update the sources. Therefore it is recommended that you install the package from a local source, allowing you to edit the code in-place. This builds the extension in place and creates a link to the development directory (see the pip docs).

```bash
pip install --verbose --no-build-isolation --editable .[all,dev,docs]
```

> You can check if the package is linked to your local clone of `f3dasm` by running `pip show list` and look for `f3dasm`.

If you want to contribute to this project, please read the [contribution guidelines](https://github.com/bessagroup/f3dasm/blob/main/CONTRIBUTING.md) and go for more instruction to the [GitHub wiki page](https://github.com/bessagroup/f3dasm/wiki).