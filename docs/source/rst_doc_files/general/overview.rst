.. _overview:

Overview
========

.. _design-of-experiments: https://bessagroup.github.io/f3dasm/classes/design/experimentdata.html
.. _sampling: https://bessagroup.github.io/f3dasm/classes/sampling/sampling.html
.. _optimizing: https://bessagroup.github.io/f3dasm/classes/optimization/optimizers.html
.. _parallelizing: <URL for parallelizing>
.. _TORQUE system: https://hpc-wiki.info/hpc/Torque

Summary
^^^^^^^

The process of structural and materials design involves a continuous search for the most efficient design based on specific criteria. 
The different boundary conditions applied to the material or structure can result in vastly different optimal compositions. 
In particular, the design of materials is faced with a high-dimensional solution space due to the overwhelming number of potential combinations that can create distinct materials. 
Unfortunately, this extensive design space poses a significant challenge to accelerate the optimization process. 
The immense number of possibilities makes it impractical to conduct experimental investigations of each concept. 
As a result, data-driven computational analyses have been a method of interest for efficiently traversing these design spaces.

Most applied fields such as Mechanical Engineering have remained conservative when it comes to openly sharing databases and software.
This is in sharp contrast to the Computer Science community, especially within the sub-field of machine learning. 
Understandably, the entry barrier is higher for researchers and practitioners from applied fields that are not specialized in software development. 
In this work we developed a general and user-friendly data-driven package for researchers and practitioners working on design and analysis of materials and structures. 
The package is called :mod:`f3dasm` (framework for data-driven design & analysis of structures and materials) and it aims at democratizing the data-driven process and making it easier to replicate research articles in this field, as well as sharing new work with the community. 
This work generalizes the original closed-source framework proposed by the senior author and his co-workers [1]_, making it more flexible and adaptable to different applications, allowing to integrate different choices of software packages needed in the different steps of the data-driven process: (1) design of experiments; (2) data generation; (3) machine learning; and (4) optimization.

Statement of need
^^^^^^^^^^^^^^^^^

The effectiveness of the first published version of :mod:`f3dasm` framework has been demonstrated in various computational mechanics and materials studies, 
such as the design of a super-compressible meta-material [2]_ and a spiderweb nano-mechanical resonator inspired 
by nature and guided by machine learning [3]_. 

The :mod:`f3dasm` framework is aimed at researchers who seek to develop such data-driven methods for structural & material modeling by incorporating modern machine learning tools. The framework integrates the following fields:

- Design of experiments, in which input variables describing the microstructure, structure, properties and external conditions of the system to be evaluated are determined and sampled;
- Data generation, typically through computational analysis, resulting in the creation of a material response database;
- Machine learning, in which a surrogate model is trained to fit experimental findings;
- Optimization, where we try to iteratively improve the model to obtain a superior design.

:mod:`f3dasm` is designed as a *modular* framework, consisting of both core and extended features, allowing user-specific instantiations of one or more steps from the above list.

The *core* functionality of the framework comprises the following features:

- provide a way to parametrize experiments with the design-of-experiments functionalities;
- enabling experiment exploration by means of sampling and design optimization;
- provide the user tools for parallelizing their program and ordering their data;
- Allowing users to deploy experiments on high-performance computer systems (TORQUE Resource Manager).

The *extensions* can be installed on the fly and contain the following features:

- provide various implementations to accommodate common data-driven workflows;
- adapter classes that link popular machine learning libraries to be used as implementations in :mod:`f3dasm`.

The Python package includes extensive implementations for each of the provided fields and are organized in an object-oriented way.

.. [1] Bessa, M. A., Bostanabad, R., Liu, Z., Hu, A., Apley, D. W., Brinson, C., Chen, W., & Liu, W. K. (2017). 
        *A framework for data-driven analysis of materials under uncertainty: Countering the curse of dimensionality. 
        Computer Methods in Applied Mechanics and Engineering*, 320, 633-667.

.. [2] Bessa, M. A., Glowacki, P., & Houlder, M. (2019). 
        *Bayesian machine learning in metamaterial design: 
        Fragile becomes supercompressible*. Advanced Materials, 31(48), 1904845.

.. [3] Shin, D., Cupertino, A., de Jong, M. H., Steeneken, P. G., Bessa, M. A., & Norte, R. A. (2022). 
        *Spiderweb nanomechanical resonators via bayesian optimization: inspired by nature and guided by machine learning*. Advanced Materials, 34(3), 2106248.


























.. The use of state-of-the-art machine learning tools for innovative structural and materials design has demonstrated their potential in various studies. 
.. Although the specific applications may differ, the data-driven modelling and optimization process remains the same. 
.. Therefore, the framework for data-driven design and analysis of structures and materials (:mod:`f3dasm`) is an attempt to develop a systematic approach of inverting the material design process. 


.. The framework, originally proposed by Bessa et al. :cite:p:`Bessa2017` integrates the following fields:

.. - **Design \& Sampling**, in which input variables describing the microstructure, structure, properties and external conditions of the system to be evaluated are determined and sampled.
.. - **Simulation**, typically through computational analysis, resulting in the creation of a material response database.
.. - **Machine learning**, in which a surrogate model is trained to fit experimental findings.
.. - **Optimization**, where we try to iteratively improve the model to obtain a superior design.

.. The effectiveness of the first published version of :mod:`f3dasm` framework has been demonstrated in various computational mechanics and materials studies, 
.. such as the design of a super-compressible meta-material :cite:p:`Bessa2019` and a spiderweb nano-mechanical resonator inspired 
.. by nature and guided by machine learning :cite:p:`Shin2022`. 


.. .. [3] Bessa, M. A., Bostanabad, R., Liu, Z., Hu, A., Apley, D. W., Brinson, C., Chen, W., & Liu, W. K. (2017). 
..         *A framework for data-driven analysis of materials under uncertainty: Countering the curse of dimensionality. 
..         Computer Methods in Applied Mechanics and Engineering*, 320, 633-667.

.. Modularity and use cases
.. ^^^^^^^^^^^^^^^^^^^^^^^^

.. The package contains a lot of implementation for each of the blocks.
.. However, the installation :mod:`f3dasm` is modular: you decide what you
.. want to use or not.

.. We can distinguish 3 ways of using :mod:`f3dasm`:

.. Using :mod:`f3dasm` to handle your design of experiments
.. -----------------------------------------------------

.. The :mod:`f3dasm` package: contains the minimal installation to use
.. :mod:`f3dasm` without extended features. 

.. .. note::

..     You can install the core package with ``pip install f3dasm`` or `read the installation instructions <https://bessagroup.github.io/f3dasm/general/gettingstarted.html>`__!

.. The core package contains the following features:

.. 1. provide a way to parametrize your experiment with the `design-of-experiments`_ classes.
.. 2. provide the option to investigate their experiment by `sampling`_ and `optimizing`_ their design.
.. 3. provide the user guidance in `parallelizing`_ their program and ordering their data.
.. 4. give the user ways of deploying their experiment at a high-performance computer system (`TORQUE system`_).

.. The core package requires the following dependencies:

.. - `numpy <https://numpy.org/doc/stable/index.html>`_ and `scipy <https://docs.scipy.org/doc/scipy/reference/>`_: for numerical operations
.. - `pandas <https://pandas.pydata.org/docs/>`_ and `SALib <https://salib.readthedocs.io/en/latest/>`_: for the representation of the design of experiments
.. - `matplotlib <https://matplotlib.org/stable/contents.html>`_: for plotting
.. - `hydra-core <https://hydra.cc/docs/intro/>`_: for deploying your experiment
.. - `pathos <https://pathos.readthedocs.io/en/latest/>`_: for multiprocessing
.. - `autograd <https://github.com/HIPS/autograd>`_: for computing gradients


.. Using :mod:`f3dasm` extended capabilities
.. --------------------------------------

.. Use existing implementations to benchmark parts of the data-driven machine learning process!

.. For this purpose, you can solely use the core package, but it is advised
.. to enrich :mod:`f3dasm` with its **extension libraries**

.. The extensions contain the following features:

.. 1. provide various **implementations** to accommodate common machine learning workflows.
.. 2. provide **adapter** classes that link common machine learning libraries to :mod:`f3dasm` base classes.

.. The following extensions libraries are available:

.. - `f3dasm_simulate <https://github.com/bessagroup/f3dasm_simulate>`_: containing various simulators ported to be used with :mod:`f3dasm`.
.. -  `f3dasm_optimize <https://github.com/bessagroup/f3dasm_optimize>`_: containing various optimizers from `GPyOpt <https://gpyopt.readthedocs.io/en/latest/>`_, `pygmo <https://esa.github.io/pygmo2/index.html>`_ and `tensorflow <https://www.tensorflow.org/api_docs/>`_

.. The main takeaway is that if your design-of-experiments is modified to
.. use the ``f3dasm.ExperimentData`` class, you are able to seamlessly
.. incorporate the extension into your application!

.. Abstraction
.. ^^^^^^^^^^^

.. By abstracting away the details of specific implementations, users and developers can better organize and reuse their code, 
.. making it easier to understand, modify, and share with others. Within the :mod:`f3dasm` framework, abstraction is done in four levels:

.. - **block**: blocks represent one of the high-level stages that can be used in the framework, e.g. the :mod:`~f3dasm.optimization` submodule. They can be put in any specific order, and incorporate a core action undertaken by the design.
.. - **base**: bases represent an abstract class of an element in the block, e.g. the :class:`~f3dasm.optimization.optimizer.Optimizer` class. Base classes are used to create a unified interface for specific implementations and are inherited from blocks.
.. - **implementation**: implementations are application of a base class feature, e.g. the :class:`~f3dasm.optimization.adam.Adam` optimizer. These can be self-coded or ported from other Python libraries.
.. - **experiment**: experiments represent executable programs that uses a certain order of blocks and specific implementations to generate results.

.. .. image:: ../../img/f3dasm-blocks.svg


.. Overview of implementations and base classes
.. --------------------------------------------

.. ===================== =============================== ========================================================================== =======================================================
.. Block                 Submodule                       Base                                                                       Implementations
.. ===================== =============================== ========================================================================== =======================================================
.. Design of Experiments :mod:`~f3dasm.design`           :class:`~f3dasm.design.design.Domain`                                 
..                       :mod:`~f3dasm.sampling`         :class:`~f3dasm.sampling.sampler.Sampler`                                  :ref:`List of samplers <implemented samplers>`
.. Data generation       :mod:`~f3dasm.datageneration`   :class:`~f3dasm.datageneration.DataGenerator`                              :ref:`List of datagenerators <implemented datagenerators>`
.. Machine learning      :mod:`~f3dasm.machinelearning`  :class:`~f3dasm.machinelearning.model.Model`                               :ref:`List of models <implemented models>`
.. Optimization          :mod:`~f3dasm.optimization`     :class:`~f3dasm.optimization.optimizer.Optimizer`                          :ref:`List of optimizers <implemented optimizers>`
.. ===================== =============================== ========================================================================== =======================================================

.. Overview of other classes
.. -------------------------

.. =============================================================== ===================================================================================
.. Class                                                           Short description
.. =============================================================== ===================================================================================
.. :class:`~f3dasm.ExperimentData`           Datastructure denoting samples from a design-of-experiments                                     
.. :class:`~f3dasm.functions.function.Function`                    Class that represents an analytical function used for benchmarking
.. :class:`~f3dasm.functions.adapters.augmentor.Augmentor`         Class that can be used to manipulate data for data-augmentation
.. :class:`~f3dasm.optimization.optimizer.OptimizerParameters`     Class that represents the hyper-parameters for a particular optimizer
.. :class:`~f3dasm.run_optimization.OptimizationResult`            Class used to store optimization results for several epochs
.. =============================================================== ===================================================================================


.. References
.. ----------

.. .. bibliography::