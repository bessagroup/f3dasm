.. image:: img/f3dasm-logo.png

Summary
-------

Welcome to the documentation page of the 'Framework for Data-Driven Design and Analysis of Structures and Materials' (:code:`f3dasm`).
Here you will find all information on installing, using and contributing to the Python package.

Statement of Need
-----------------

The use of state-of-the-art machine learning tools for innovative structural and materials design has demonstrated their potential in various studies. 
Although the specific applications may differ, the data-driven modelling and optimization process remains the same. 
Therefore, the framework for data-driven design and analysis of structures and materials (:code:`f3dasm`) is an attempt to develop a systematic approach of inverting the material design process. 


The framework, originally proposed by Bessa et al. [1]_ integrates the following fields:

- **Design \& Sampling**, in which input variables describing the microstructure, structure, properties and external conditions of the system to be evaluated are determined and sampled.
- **Simulation**, typically through computational analysis, resulting in the creation of a material response database.
- **Machine learning**, in which a surrogate model is trained to fit experimental findings.
- **Optimization**, where we try to iteratively improve the model to obtain a superior design.

The effectiveness of the first published version of :code:`f3dasm` framework has been demonstrated in various computational mechanics and materials studies, 
such as the design of a super-compressible meta-material [2]_ and a spiderweb nano-mechanical resonator inspired 
by nature and guided by machine learning [3]_. 

.. [1] Bessa, M. A., Bostanabad, R., Liu, Z., Hu, A., Apley, D. W., Brinson, C., Chen, W., & Liu, W. K. (2017). 
        *A framework for data-driven analysis of materials under uncertainty: Countering the curse of dimensionality. 
        Computer Methods in Applied Mechanics and Engineering*, 320, 633-667.

.. [2] Bessa, M. A., Glowacki, P., & Houlder, M. (2019). 
        *Bayesian machine learning in metamaterial design: 
        Fragile becomes supercompressible*. Advanced Materials, 31(48), 1904845.

.. [3] Shin, D., Cupertino, A., de Jong, M. H., Steeneken, P. G., Bessa, M. A., & Norte, R. A. (2022). 
        *Spiderweb nanomechanical resonators via bayesian optimization: inspired by nature and guided by machine learning*. Advanced Materials, 34(3), 2106248.

Authorship & Citation
---------------------

:code:`f3dasm` was originally created by Miguel A. Bessa [1]_, and is now maintained by Martin van der Schelling [2]_.

.. [1] Associate Professor, Brown University.

.. [2] PhD Candiate, Delft University of Technology, `Website <https://mpvanderschelling.github.io/>`_ , `GitHub <https://github.com/mpvanderschelling/>`_

If you use :code:`f3dasm` in your research or in a scientific publication, it is appreciated that you cite the paper below:

**Computer Methods in Applied Mechanics and Engineering** (`paper <https://doi.org/10.1016/j.cma.2017.03.037>`_):

.. code-block:: tex

    @article{Bessa2017,
    title={A framework for data-driven analysis of materials under uncertainty: Countering the curse of dimensionality},
    author={Bessa, Miguel A and Bostanabad, Ramin and Liu, Zeliang and Hu, Anqi and Apley, Daniel W and Brinson, Catherine and Chen, Wei and Liu, Wing Kam},
    journal={Computer Methods in Applied Mechanics and Engineering},
    volume={320},
    pages={633--667},
    year={2017},
    publisher={Elsevier}
    }

Getting started
---------------


The best way to get started is to follow the `installation instructions <https://bessagroup.github.io/f3dasm/general/gettingstarted.html>`_



Useful links
------------

* `GitHub repository <https://github.com/bessagroup/F3DASM/tree/main>`_ (source code)
* `Wiki for development <https://github.com/bessagroup/F3DASM/wiki>`_
* `PyPI package <https://pypi.org/project/f3dasm/>`_ (distribution package)

Related extension libraries
---------------------------
* `f3dasm_optimize <https://github.com/bessagroup/f3dasm_optimize>`_: Optimization algorithms for the `f3dasm` package.
* `f3dasm_simulate <https://github.com/bessagroup/f3dasm_optimize>`_: Simulators for the `f3dasm` package.
* `f3dasm_teach <https://github.com/mpvanderschelling/f3dasm_teach>`_: Hub for practical session and educational material on using `f3dasm`.

License
-------
Copyright 2023, Martin van der Schelling

All rights reserved.

:code:`f3dasm` is a free and open-source software published under a `BSD 3-Clause License <https://github.com/bessagroup/f3dasm/blob/main/LICENSE>`_.
