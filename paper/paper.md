---
title: 'f3dasm: Framework for Data-Driven Design & Analysis of Structures & Materials'
tags:
  - Python
  - data-driven
  - materials
  - framework
  - machine learning
authors:
  - name: Martin van der Schelling
    orcid: 0000-0003-3602-0452
    affiliation: 1
  - name: Bernardo P. Ferreira
    orcid: 0000-0001-5956-3877
    affiliation: 2
  - name: Miguel A. Bessa
    corresponding: true
    orcid: 0000-0002-6216-0355
    affiliation: 2
affiliations:
 - name: Materials Science & Engineering, Delft University of Technology, the Netherlands
   index: 1
 - name: School of Engineering, Brown University, United States of America
   index: 2
date: 31 May 2024
bibliography: paper.bib

---

# Summary

[`f3dasm`](https://github.com/bessagroup/f3dasm) (Framework for Data-driven Design and Analysis of Structures \& Materials) is a Python project that provides a general and user-friendly data-driven framework for researchers and practitioners working on the design and analysis of materials and structures. The package aims to streamline the data-driven process and make it easier to replicate research articles in this field, as well as share new work with the community. 

![Logo of [`f3dasm`](https://github.com/bessagroup/f3dasm). \label{fig:f3dasm_logo}](f3dasm_logo_long.png)

# Statement of need

In the last decades, advancements in computational resources have accelerated novel inverse design approaches for structures and materials. In particular, data-driven methods leveraging machine learning techniques play a major role in shaping our design processes today.

Constructing a large material response database poses practical challenges, such as proper data management, efficient parallel computing, and integration with third-party software. Because most applied fields remain conservative when it comes to openly sharing databases and software, a lot of research time is instead being allocated to implement common procedures that would be otherwise readily available. This lack of shared practices also leads to compatibility issues for benchmarking and replication of results by violating the FAIR principles.

In this work we introduce an interface for researchers and practitioners working on the design and analysis of materials and structures. The package is called [`f3dasm`](https://github.com/bessagroup/f3dasm) (Framework for Data-driven Design \& Analysis of Structures and Materials). This work generalizes the original closed-source framework proposed by the Bessa and co-workers [@Bessa2017], making it more flexible and adaptable to different applications, namely by allowing the integration of different choices of software packages needed in the different steps of the data-driven process:

- **Design of experiments**, in which input variables describing the microstructure, properties and external conditions of the system are determined and sampled;
- **Data generation**, typically through computational analyses, resulting in the creation of a material response database [@Ferreira2023];
- **Machine learning**, in which a surrogate model is trained to fit experimental findings;
- **Optimization**, where we try to iteratively improve the design.

\autoref{fig:data-driven-process} provides an illustration of the stages in the data-driven process. 

![Illustration of the `f3dasm` data-driven process. \label{fig:data-driven-process}](data-driven-process.png)

[`f3dasm`](https://github.com/bessagroup/f3dasm) is an [open-source Python package](https://pypi.org/project/f3dasm/) compatible with Python 3.8 or later. The library includes a suite of benchmark functions, optimization algorithms, and sampling strategies to serve as default implementations. Furthermore, [`f3dasm`](https://github.com/bessagroup/f3dasm) offers automatic data management for experiments, easy integration with high-performance computing systems, and compatibility with the hydra configuration manager. Comprehensive [online documentation](https://f3dasm.readthedocs.io/en/latest/) is also available to assist users and developers of the framework.

In a similar scope, it is worth mentioning the projects [simmate](https://github.com/jacksund/simmate) [@Sundberg2022] and [strucscan](https://github.com/ICAMS/strucscan), as they provide tools for the management of materials science simulation and databases. However, these projects focus on the generation and retrieval of materials properties and do not include machine learning or optimization interfaces. In recent years, numerous optimization frameworks have been developed to facilitate data-driven design. [Optuna](https://optuna.org/) is a hyperparameter optimization framework that combines a variety of optimization algorithms with dynamically constructed search space [@Akiba2019] and [pygmo](https://github.com/esa/pagmo2) provides unified interfaces for parallel global optimization [@Biscani2020]. Interfaces to these and many other optimization frameworks have been integrated into a separate package [`f3dasm_optimize`](https://github.com/bessagroup/f3dasm_optimize), and can be used in conjunction with [`f3dasm`](https://github.com/bessagroup/f3dasm).


# Acknowledgements

We would express our gratitude to Jiaxiang Yi for his contributions to writing an interface with the ABAQUS simulation software.

# References
