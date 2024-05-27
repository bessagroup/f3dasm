---
title: 'f3dasm: Framework for Data-Driven Design & Analysis of Structures & Materials'
tags:
  - Python
  - data-driven
  - materials
  - framework
  - machine learning
authors:
  - name: Martin P. van der Schelling
    orcid: 0000-0003-3602-0452
    # equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
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
<!-- A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience. -->

[`f3dasm`](https://github.com/bessagroup/f3dasm) (Framework for Data-driven Design and Analysis of Structures \& Materials) is a Python project that provides a general and user-friendly data-driven framework for researchers and practitioners working on the design and analysis of materials and structures. The package aims to streamline the data-driven process and make it easier to replicate research articles in this field, as well as share new work with the community. 

![Logo of [`f3dasm`](https://github.com/bessagroup/f3dasm). \label{fig:f3dasm_logo}](f3dasm_logo.png)

# Statement of need
<!-- A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work. -->

In the last decades, advancements in computational resources have accelerated novel inverse design approaches for structures and materials. In particular data-driven methods leveraging machine learning techniques play a major role in shaping our design processes today.

Constructing a large material response database poses practical challenges, such as proper data management, efficient parallel computing and integration with third-party software. Because most applied fields remain conservative when it comes to openly sharing databases and software, a lot of research time is instead being allocated to implement common procedures that would be otherwise readily available. This lack of shared practices also leads to compatibility issues for benchmarking and replication of results by violating the FAIR principles.

In this work we introduce an interface for researchers and practitioners working on design and analysis of materials and structures. The package is called [`f3dasm`](https://github.com/bessagroup/f3dasm) (Framework for Data-driven Design \& Analysis of Structures and Materials) This work generalizes the original closed-source framework proposed by the Bessa and co-workers [@Bessa2017], making it more flexible and adaptable to different applications, namely by allowing the integration of different choices of software packages needed in the different steps of the data-driven process: (1) design of experiments; (2) data generation; (3) machine learning; and (4) optimization. \autoref{fig:data-driven-process} provides an illustration of the stages in the data-driven process. 

![Illustration of the data-driven process. \label{fig:data-driven-process}](data-driven-process.png)

# Acknowledgements

We would express our gratitude to Jiaxiang Yi for his contributions to writing an interface with the ABAQUS simulation software.

# References