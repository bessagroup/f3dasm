.. _overview:

Overview
========

A quick overview of the f3dasm package.

----

Conceptual framework
--------------------

``f3dasm`` is a Python project that provides a general and user-friendly data-driven framework for researchers and practitioners working on the design and analysis of materials and structures [1]_. 
The package aims to streamline the data-driven process and make it easier to replicate research articles in this field, as well as share new work with the community. 

In the last decades, advancements in computational resources have accelerated novel inverse design approaches for structures and materials. 
In particular data-driven methods leveraging machine learning techniques play a major role in shaping our design processes today.

Constructing a large material response database poses practical challenges, such as proper data management, efficient parallel computing and integration with third-party software. 
Because most applied fields remain conservative when it comes to openly sharing databases and software, a lot of research time is instead being allocated to implement common procedures that would be otherwise readily available. 
This lack of shared practices also leads to compatibility issues for benchmarking and replication of results by violating the FAIR principles.

In this work we introduce an interface for researchers and practitioners working on design and analysis of materials and structures. 
The package is called ``f3dasm`` (Framework for Data-driven Design \& Analysis of Structures and Materials).
This work generalizes the original closed-source framework proposed by the Bessa and co-workers [2]_, making it more flexible and adaptable to different applications, 
namely by allowing the integration of different choices of software packages needed in the different steps of the data-driven process:

- **Design of experiments**, in which input variables describing the microstructure, properties and external conditions of the system are determined and sampled.
- **Data generation**, typically through computational analyses, resulting in the creation of a material response database.
- **Machine learning**, in which a surrogate model is trained to fit experimental findings.
- **Optimization**, where we try to iteratively improve the design

.. image:: ../../img/data-driven-process.png
    :align: center
    :width: 100%

|

----

Computational framework
-----------------------

``f3dasm`` is an `open-source Python package <https://pypi.org/project/f3dasm/>`_ compatible with Python 3.8 or later. Some of the key features are:

-  Modular design 

    - The framework introduces flexible interfaces, allowing users to easily integrate their own models and algorithms.

- Automatic data management

    -  The framework automatically manages I/O processes, saving you time and effort implementing these common procedures.

- Easy parallelization

    - The framework manages parallelization of experiments, and is compatible with both local and high-performance cluster computing.

- Built-in defaults

    - The framework includes a collection of :ref:`benchmark functions <implemented-benchmark-functions>`, :ref:`optimization algorithms <implemented optimizers>` and :ref:`sampling strategies <implemented samplers>` to get you started right away!

- Hydra integration

    - The framework is integrated with `hydra <https://hydra.cc/>`_ configuration manager, to easily manage and run experiments.

Comprehensive `online documentation <https://f3dasm.readthedocs.io/en/latest/>`_ is also available to assist users and developers of the framework.


.. [1] van der Schelling, M. P., Ferreira, B. P., & Bessa, M. A. (2024). 
        *f3dasm: Framework for data-driven design and analysis of structures and materials. Journal of Open Source Software*, 9(100), 6912.

.. [2] Bessa, M. A., Bostanabad, R., Liu, Z., Hu, A., Apley, D. W., Brinson, C., Chen, W., & Liu, W. K. (2017). 
        *A framework for data-driven analysis of materials under uncertainty: Countering the curse of dimensionality. 
        Computer Methods in Applied Mechanics and Engineering*, 320, 633-667.
