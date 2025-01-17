f3dasm
------

<div align="center"><img src="https://raw.githubusercontent.com/bessagroup/f3dasm/main/logo.png" width="800"/></div>

***

[![DOI](https://joss.theoj.org/papers/b0a25f75a32ae95a0a75bf3118952a5d/status.svg)](https://joss.theoj.org/papers/b0a25f75a32ae95a0a75bf3118952a5d)
[![Python](https://img.shields.io/pypi/pyversions/f3dasm)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/f3dasm.svg)](https://pypi.org/project/f3dasm/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/f3dasm.svg)](https://anaconda.org/conda-forge/f3dasm)
[![GitHub license](https://img.shields.io/badge/license-BSD-blue)](https://github.com/bessagroup/f3dasm)
[![Documentation Status](https://readthedocs.org/projects/f3dasm/badge/?version=latest)](https://f3dasm.readthedocs.io/en/latest/?badge=latest)

[**Docs**](https://f3dasm.readthedocs.io/)
| [**Installation**](https://f3dasm.readthedocs.io/en/latest/rst_doc_files/general/installation.html)
| [**GitHub**](https://github.com/bessagroup/f3dasm)
| [**PyPI**](https://pypi.org/project/f3dasm/)
| [**Conda**](https://anaconda.org/conda-forge/f3dasm)
| [**Paper**](https://doi.org/10.21105/joss.06912)

## Summary

Welcome to `f3dasm`, a **f**ramework for **d**ata-**d**riven **d**esign and **a**nalysis of **s**tructures and **m**aterials.

`f3dasm` introduces a general and user-friendly data-driven Python package for researchers and practitioners working on design and analysis of materials and structures. Some of the key features include:

-  **Modular design** 
    - The framework introduces flexible interfaces, allowing users to easily integrate their own models and algorithms.

- **Automatic data management**
    -  The framework automatically manages I/O processes, saving you time and effort implementing these common procedures.

- **Easy parallelization**
    - The framework manages parallelization of experiments, and is compatible with both local and high-performance cluster computing.

- **Built-in defaults**
    - The framework includes a collection of benchmark functions, optimization algorithms and sampling strategies to get you started right away!

- **Hydra integration**
    - The framework is supports the [hydra](https://hydra.cc/) configuration manager, to easily manage and run experiments.


## Getting started

`f3dasm` is available at [the Python Package Index]() and on [Anaconda Cloud](https://anaconda.org/conda-forge/f3dasm). To get started:

```bash
# PyPI
$ pip install f3dasm
```

or

```bash
# PyPI
$ conda install conda-forge::f3dasm
```

* Follow the complete [installation instructions](https://f3dasm.readthedocs.io/en/latest/rst_doc_files/general/installation.html) to get going!
* Read the [overview](https://f3dasm.readthedocs.io/en/latest/rst_doc_files/general/installation.html) section, containing a brief introduction to the framework and a statement of need.
* Check out the [tutorials](https://f3dasm.readthedocs.io/en/latest/auto_examples/index.html) section, containing a collection of examples to get you familiar with the framework.

## Illustrative benchmarks

This package includes a collection of illustrative benchmark studies that demonstrate the capabilities of the framework. These studies are available in the `/studies/` folder, and include the following studies:

- Benchmarking optimization algorithms against well-known benchmark functions
- 'Fragile Becomes Supercompressible' ([Bessa et al. (2019)](https://onlinelibrary.wiley.com/doi/full/10.1002/adma.201904845))

## Authorship & Citation

Current creator and developer: M.P. van der Schelling<sup>[1](#f1)</sup>

<sup id="f1"> 1 </sup>Doctoral Researcher in Materials Science and Engineering, Delft University of Technology: [ORCID](https://orcid.org/0000-0003-3602-0452), [Website](https://github.com/mpvanderschelling/)

If you use `f3dasm` in your research or in a scientific publication, it is appreciated that you cite the paper below:

**Journal of Open Source Software** ([paper](https://doi.org/10.21105/joss.06912)):
```
@article{vanderSchelling2024,
  title = {f3dasm: Framework for Data-Driven Design and Analysis of Structures and Materials},
  author = {M. P. van der Schelling and B. P. Ferreira and M. A. Bessa},
  doi = {10.21105/joss.06912},
  url = {https://doi.org/10.21105/joss.06912},
  year = {2024},
  publisher = {The Open Journal},
  volume = {9},
  number = {100},
  pages = {6912},
  journal = {Journal of Open Source Software}
}
```

The Bessa research group at TU Delft is small... At the moment, we have limited availability to help future users/developers adapting the code to new problems, but we will do our best to help!

## Community Support

If you find any **issues, bugs or problems** with this template, please use the [GitHub issue tracker](https://github.com/bessagroup/f3dasm/issues) to report them.

## License

Copyright 2025, Martin van der Schelling

All rights reserved.

This project is licensed under the BSD 3-Clause License. See [LICENSE](https://github.com/bessagroup/f3dasm/blob/main/LICENSE) for the full license text.
