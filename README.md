f3dasm
------
*Framework for data-driven design \& analysis of structures and materials*

***

[![Python](https://img.shields.io/pypi/pyversions/f3dasm)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/f3dasm.svg)](https://pypi.org/project/f3dasm/)
[![GitHub license](https://img.shields.io/badge/license-BSD-blue)](https://github.com/bessagroup/f3dasm)
[![Documentation Status](https://readthedocs.org/projects/f3dasm/badge/?version=latest)](https://f3dasm.readthedocs.io/en/latest/?badge=latest)

[**Docs**](https://f3dasm.readthedocs.io/)
| [**Installation**](https://f3dasm.readthedocs.io/en/latest/rst_doc_files/general/gettingstarted.html)
| [**GitHub**](https://github.com/bessagroup/f3dasm)
| [**PyPI**](https://pypi.org/project/f3dasm/)

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

The best way to get started is to follow the [installation instructions](https://f3dasm.readthedocs.io/en/latest/rst_doc_files/general/gettingstarted.html).

## Illustrative benchmarks

This package includes a collection of illustrative benchmark studies that demonstrate the capabilities of the framework. These studies are available in the `/studies` folder, and include the following studies:

- Benchmarking optimization algorithms against well-known benchmark functions
- 'Fragile Becomes Supercompressible' ([Bessa et al. (2019)](https://onlinelibrary.wiley.com/doi/full/10.1002/adma.201904845))

## Authorship

* Current created and developer: [M.P. van der Schelling](https://github.com/mpvanderschelling/) (M.P.vanderSchelling@tudelft.nl)

The Bessa research group at TU Delft is small... At the moment, we have limited availability to help future users/developers adapting the code to new problems, but we will do our best to help!



## Referencing

If you use or edit our work, please cite at least one of the appropriate references:

[1] Bessa, M. A., Bostanabad, R., Liu, Z., Hu, A., Apley, D. W., Brinson, C., Chen, W., & Liu, W. K. (2017). A framework for data-driven analysis of materials under uncertainty: Countering the curse of dimensionality. Computer Methods in Applied Mechanics and Engineering, 320, 633-667.

[2] Bessa, M. A., & Pellegrino, S. (2018). Design of ultra-thin shell structures in the stochastic post-buckling range using Bayesian machine learning and optimization. International Journal of Solids and Structures, 139, 174-188.

[3] Bessa, M. A., Glowacki, P., & Houlder, M. (2019). Bayesian machine learning in metamaterial design: fragile becomes super-compressible. Advanced Materials, 31(48), 1904845.

[4] Mojtaba, M., Bostanabad, R., Chen, W., Ehmann, K., Cao, J., & Bessa, M. A. (2019). Deep learning predicts path-dependent plasticity. Proceedings of the National Academy of Sciences, 116(52), 26414-26420.

## Community Support

If you find any **issues, bugs or problems** with this template, please use the [GitHub issue tracker](https://github.com/bessagroup/f3dasm/issues) to report them.

## License

Copyright 2024, Martin van der Schelling

All rights reserved.

This project is licensed under the BSD 3-Clause License. See [LICENSE](https://github.com/bessagroup/f3dasm/blob/main/LICENSE) for the full license text.