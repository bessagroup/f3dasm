Built-in functionalities
========================

:mod:`f3dasm` provides a set of built-in functionalities that can be used to perform data-driven optimization and sensitivity analysis. 
All built-ins are implementations of the :class:`~f3dasm.Block` class and can be used on your :class:`~f3dasm.ExperimentData` object.

The built-in blocks can be initialized by either importing the functions directly from the respective submodules or by using a string argument to specify the built-in function you want to use.

=============================== ======================================= ===================================================
Part of the data-driven process Submodule for built-ins                 Function to call with string argument
=============================== ======================================= ===================================================
Sampling                        :mod:`f3dasm.design`                    :func:`f3dasm.create_sampler`
Data generation                 :mod:`f3dasm.datageneration`            :func:`f3dasm.create_datagenerator`
Optimization                    :mod:`f3dasm.optimization`              :func:`f3dasm.create_optimizer`
=============================== ======================================= ===================================================

:mod:`f3dasm` provides two ways to use the built-in functionalities:

1. Call the built-in functions

You can import the built-in functions directly from the respective submodules and call them to change the (hyper)parameters.

.. code-block:: python

    from f3dasm.design import random
    from f3dasm.datageneration import ackley

    # Call the random uniform sampler with a specific seed
    sampler_block = random(seed=123)

    # Create a 2D instance of the 'Ackley' function with its box-constraints scaled to [0, 1]
    data_generation_block = ackley(scale_bounds=[[0., 1.], [0., 1.]])

    # Create an empty Domain
    domain = Domain()

    # Add two continuous parameters 'x0' and 'x1'
    domain.add_float(name='x0', low=0.0, high=1.0)
    domain.add_float(name='x1', low=0.0, high=1.0)

    # Create an empty ExperimentData object with the domain
    experiment_data = ExperimentData(domain=domain)

    # 1. Sampling
    experiment_data = sampler_block(data=experiment_data, n_samples=10)
    
    # 2. Evaluating the samples

    data_generation_block.arm(data=experiment_data)
    experiment_data = data_generation_block.call(data=experiment_data)

2. Use a string argument

Alternatively, you can use a string argument to specify the built-in function you want to use.


.. code-block:: python
  
    from f3dasm import create_sampler, create_datagenerator

    sampler_block = create_sampler(sampler='random', seed=123)

    data_generation_block = create_datagenerator(
      data_generator='ackley',
      scale_bounds=[[0., 1.], [0., 1.]])

    # Create an empty Domain
    domain = Domain()

    # Add two continuous parameters 'x0' and 'x1'
    domain.add_float(name='x0', low=0.0, high=1.0)
    domain.add_float(name='x1', low=0.0, high=1.0)

    # Create an empty ExperimentData object with the domain
    experiment_data = ExperimentData(domain=domain)

    # 1. Sampling
    experiment_data = sampler_block(data=experiment_data, n_samples=10)
    
    # 2. Evaluating the samples

    data_generation_block.arm(data=experiment_data)
    experiment_data = data_generation_block.call(data=experiment_data)



.. warning::

  The built-in functionalities are designed with the built-in parameters in mind! 
  This means that in order to make use of the samplers, benchmark functions and optimizers, 
  you are restricted to add parameters via the :meth:`~f3dasm.design.Domain.add_float`, :meth:`~f3dasm.design.Domain.add_int`, 
  :meth:`~f3dasm.design.Domain.add_category` and :meth:`~f3dasm.design.Domain.add_constant` methods.

.. _implemented samplers:

Implemented samplers
--------------------

The following built-in implementations of samplers can be used in the data-driven process.

======================== ============================= ======================================== ===========================================================================================================
Name                     Key-word                      Function                                 Reference
======================== ============================= ======================================== ===========================================================================================================
Random Uniform sampling  ``"random"``                  :func:`~f3dasm.design.random`            `numpy.random.uniform <https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html>`_
Latin Hypercube sampling ``"latin"``                   :func:`~f3dasm.design.latin`             `SALib.latin <https://salib.readthedocs.io/en/latest/api/SALib.sample.html?highlight=latin%20hypercube#SALib.sample.latin.sample>`_
Sobol Sequence sampling  ``"sobol"``                   :func:`~f3dasm.design.sobol`             `SALib.sobol_sequence <https://salib.readthedocs.io/en/latest/api/SALib.sample.html?highlight=sobol%20sequence#SALib.sample.sobol_sequence.sample>`_
Grid Search sampling     ``"grid"``                    :func:`~f3dasm.design.grid`              `itertools.product <https://docs.python.org/3/library/itertools.html#itertools.product>`_
======================== ============================= ======================================== ===========================================================================================================

.. _implemented-benchmark-functions:

Implemented benchmark functions
-------------------------------

These benchmark functions are taken and modified from the `Python Benchmark Test Optimization Function Single Objective <https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective>`_ github repository.
The following implementations of benchmark functions can instantiated with the name in the 'Data-generator argument' column.

.. note::

  Not all benchmark functions are implemented for all dimensions. 
  If you want to use a benchmark function for a dimension that is not implemented, you will get a :class:`~NotImplementedError`.

Convex functions
^^^^^^^^^^^^^^^^

======================== ====================================================== ===============================================================
Name                     Key-word                                               Function
======================== ====================================================== ===============================================================
Ackley N. 2              ``"Ackley N. 2"``                                      :func:`~f3dasm.datageneration.functions.ackleyn2`
Bohachevsky N. 1         ``"Bohachevsky N. 1"``                                 :func:`~f3dasm.datageneration.functions.bohachevskyn1`
Booth                    ``"Booth"``                                            :func:`~f3dasm.datageneration.functions.booth`
Brent                    ``"Brent"``                                            :func:`~f3dasm.datageneration.functions.brent`
Brown                    ``"Brown"``                                            :func:`~f3dasm.datageneration.functions.brown`
Bukin N. 6               ``"Bukin N. 6"``                                       :func:`~f3dasm.datageneration.functions.bukinn6`
Dixon Price              ``"Dixon Price"``                                      :func:`~f3dasm.datageneration.functions.dixonprice`
Exponential              ``"Exponential"``                                      :func:`~f3dasm.datageneration.functions.exponential`
Matyas                   ``"Matyas"``                                           :func:`~f3dasm.datageneration.functions.matyas`
McCormick                ``"McCormick"``                                        :func:`~f3dasm.datageneration.functions.mccormick`
Powell                   ``"Powell"``                                           :func:`~f3dasm.datageneration.functions.powell`
Rotated Hyper-Ellipsoid  ``"Rotated Hyper-Ellipsoid"``                          :func:`~f3dasm.datageneration.functions.rotatedhyperellipsoid`
Schwefel 2.20            ``"Schwefel 2.20"``                                    :func:`~f3dasm.datageneration.functions.schwefel2_20`
Schwefel 2.21            ``"Schwefel 2.21"``                                    :func:`~f3dasm.datageneration.functions.schwefel2_21`
Schwefel 2.22            ``"Schwefel 2.22"``                                    :func:`~f3dasm.datageneration.functions.schwefel2_22`
Schwefel 2.23            ``"Schwefel 2.23"``                                    :func:`~f3dasm.datageneration.functions.schwefel2_23`
Sphere                   ``"Sphere"``                                           :func:`~f3dasm.datageneration.functions.sphere`
Sum Squares              ``"Sum Squares"``                                      :func:`~f3dasm.datageneration.functions.sumsquares`
Thevenot                 ``"Thevenot"``                                         :func:`~f3dasm.datageneration.functions.thevenot`
Trid                     ``"Trid"``                                             :func:`~f3dasm.datageneration.functions.trid`
======================== ====================================================== ===============================================================




Seperable functions
^^^^^^^^^^^^^^^^^^^

======================== ============================================== ==========================================================
Name                     Key-word                                       Function
======================== ============================================== ==========================================================
Ackley                   ``"Ackley"``                                   :func:`~f3dasm.datageneration.functions.ackley`
Bohachevsky N. 1         ``"Bohachevsky N. 1"``                         :func:`~f3dasm.datageneration.functions.bohachevskyn1`
Easom                    ``"Easom"``                                    :func:`~f3dasm.datageneration.functions.easom`
Egg Crate                ``"Egg Crate"``                                :func:`~f3dasm.datageneration.functions.eggcrate`
Exponential              ``"Exponential"``                              :func:`~f3dasm.datageneration.functions.exponential`
Griewank                 ``"Griewank"``                                 :func:`~f3dasm.datageneration.functions.griewank`
Michalewicz              ``"Michalewicz"``                              :func:`~f3dasm.datageneration.functions.michalewicz`
Powell                   ``"Powell"``                                   :func:`~f3dasm.datageneration.functions.powell`
Qing                     ``"Qing"``                                     :func:`~f3dasm.datageneration.functions.qing`
Quartic                  ``"Quartic"``                                  :func:`~f3dasm.datageneration.functions.quartic`
Rastrigin                ``"Rastrigin"``                                :func:`~f3dasm.datageneration.functions.rastrigin`
Schwefel                 ``"Schwefel"``                                 :func:`~f3dasm.datageneration.functions.schwefel`
Schwefel 2.20            ``"Schwefel 2.20"``                            :func:`~f3dasm.datageneration.functions.schwefel2_20`
Schwefel 2.21            ``"Schwefel 2.21"``                            :func:`~f3dasm.datageneration.functions.schwefel2_21`
Schwefel 2.22            ``"Schwefel 2.22"``                            :func:`~f3dasm.datageneration.functions.schwefel2_22`
Schwefel 2.23            ``"Schwefel 2.23"``                            :func:`~f3dasm.datageneration.functions.schwefel2_23`
Sphere                   ``"Sphere"``                                   :func:`~f3dasm.datageneration.functions.sphere`
Styblinski Tank          ``"Styblinski Tank"``                          :func:`~f3dasm.datageneration.functions.styblinskitang`
Sum Squares              ``"Sum Squares"``                              :func:`~f3dasm.datageneration.functions.sumsquares`
Thevenot                 ``"Thevenot"``                                 :func:`~f3dasm.datageneration.functions.thevenot`
Xin She Yang             ``"Xin She Yang"``                             :func:`~f3dasm.datageneration.functions.xin_she_yang`
======================== ============================================== ==========================================================

Multimodal functions
^^^^^^^^^^^^^^^^^^^^

======================== ================================================ ===========================================================
Name                     Key-word                                         Function
======================== ================================================ ===========================================================
Ackley                   ``"Ackley"``                                     :func:`~f3dasm.datageneration.functions.ackley`
Ackley N. 3              ``"Ackley N. 3"``                                :func:`~f3dasm.datageneration.functions.ackleyn3`
Ackley N. 4              ``"Ackley N. 4"``                                :func:`~f3dasm.datageneration.functions.ackleyn4`
Adjiman                  ``"Adjiman"``                                    :func:`~f3dasm.datageneration.functions.adjiman`
Bartels                  ``"Bartels"``                                    :func:`~f3dasm.datageneration.functions.bartels`
Beale                    ``"Beale"``                                      :func:`~f3dasm.datageneration.functions.beale`
Bird                     ``"Bird"``                                       :func:`~f3dasm.datageneration.functions.bird`
Bohachevsky N. 2         ``"Bohachevsky N. 2"``                           :func:`~f3dasm.datageneration.functions.bohachevskyn2`
Bohachevsky N. 3         ``"Bohachevsky N. 3"``                           :func:`~f3dasm.datageneration.functions.bohachevskyn3`
Branin                   ``"Branin"``                                     :func:`~f3dasm.datageneration.functions.branin`
Bukin N. 6               ``"Bukin N. 6"``                                 :func:`~f3dasm.datageneration.functions.bukinn6`
Colville                 ``"Colville"``                                   :func:`~f3dasm.datageneration.functions.colville`
Cross-in-Tray            ``"Cross-in-Tray"``                              :func:`~f3dasm.datageneration.functions.crossintray`
De Jong N. 5             ``"De Jong N. 5"``                               :func:`~f3dasm.datageneration.functions.dejongn5`
Deckkers-Aarts           ``"Deckkers-Aarts"``                             :func:`~f3dasm.datageneration.functions.deckkersaarts`
Easom                    ``"Easom"``                                      :func:`~f3dasm.datageneration.functions.easom`
Egg Crate                ``"Egg Crate"``                                  :func:`~f3dasm.datageneration.functions.eggcrate`
Egg Holder               ``"Egg Holder"``                                 :func:`~f3dasm.datageneration.functions.eggholder`
Goldstein-Price          ``"Goldstein-Price"``                            :func:`~f3dasm.datageneration.functions.goldsteinprice`
Happy Cat                ``"Happy Cat"``                                  :func:`~f3dasm.datageneration.functions.happycat`
Himmelblau               ``"Himmelblau"``                                 :func:`~f3dasm.datageneration.functions.himmelblau`
Holder-Table             ``"Holder-Table"``                               :func:`~f3dasm.datageneration.functions.holdertable`
Keane                    ``"Keane"``                                      :func:`~f3dasm.datageneration.functions.keane`
Langermann               ``"Langermann"``                                 :func:`~f3dasm.datageneration.functions.langermann`
Levy                     ``"Levy"``                                       :func:`~f3dasm.datageneration.functions.levy`
Levy N. 13               ``"Levy N. 13"``                                 :func:`~f3dasm.datageneration.functions.levyn13`
McCormick                ``"McCormick"``                                  :func:`~f3dasm.datageneration.functions.mccormick`
Michalewicz              ``"Michalewicz"``                                :func:`~f3dasm.datageneration.functions.michalewicz`
Periodic                 ``"Periodic"``                                   :func:`~f3dasm.datageneration.functions.periodic`
Qing                     ``"Qing"``                                       :func:`~f3dasm.datageneration.functions.qing`
Quartic                  ``"Quartic"``                                    :func:`~f3dasm.datageneration.functions.quartic`
Rastrigin                ``"Rastrigin"``                                  :func:`~f3dasm.datageneration.functions.rastrigin`
Rosenbrock               ``"Rosenbrock"``                                 :func:`~f3dasm.datageneration.functions.rosenbrock`
Salomon                  ``"Salomon"``                                    :func:`~f3dasm.datageneration.functions.salomon`
Schwefel                 ``"Schwefel"``                                   :func:`~f3dasm.datageneration.functions.schwefel`
Shekel                   ``"Shekel"``                                     :func:`~f3dasm.datageneration.functions.shekel`
Shubert                  ``"Shubert"``                                    :func:`~f3dasm.datageneration.functions.shubert`
Shubert N. 3             ``"Shubert N. 3"``                               :func:`~f3dasm.datageneration.functions.shubertn3`
Shubert N. 4             ``"Shubert N. 4"``                               :func:`~f3dasm.datageneration.functions.shubertn4`
Styblinski Tank          ``"Styblinski Tank"``                            :func:`~f3dasm.datageneration.functions.styblinskitang`
Thevenot                 ``"Thevenot"``                                   :func:`~f3dasm.datageneration.functions.thevenot`
Xin She Yang             ``"Xin She Yang"``                               :func:`~f3dasm.datageneration.functions.xin_she_yang`
======================== ================================================ ===========================================================


.. _implemented optimizers:

Implemented optimizers
----------------------

The following implementations of optimizers can found under the :mod:`f3dasm.optimization` module: 
These are ported from `scipy-optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_

======================== ========================================================================= ============================================== ===========================================================================================================
Name                     Key-word                                                                  Function                                        Reference
======================== ========================================================================= ============================================== ===========================================================================================================
Conjugate Gradient       ``"cg"``                                                                  :func:`~f3dasm.optimization.cg`                `scipy.minimize CG <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cg.html>`_
L-BFGS-B                 ``"lbfgsb"``                                                              :func:`~f3dasm.optimization.lbfgsb`            `scipy.minimize L-BFGS-B <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_
Nelder Mead              ``"nelder_mead"``                                                         :func:`~f3dasm.optimization.nelder_mead`       `scipy.minimize NelderMead <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html>`_
Random search            ``"random_search"``                                                       :func:`~f3dasm.optimization.random_search`     `numpy <https://numpy.org/doc/>`_
======================== ========================================================================= ============================================== ===========================================================================================================

.. _f3dasm-optimize:

:code:`f3dasm-optimize`
^^^^^^^^^^^^^^^^^^^^^^^

The :mod:`f3dasm.optimization` module is designed to be easily extended by third-party libraries.
These extensions are provided as separate package: `f3dasm_optimize <https://github.com/bessagroup/f3dasm_optimize>`_, which can be installed via pip:

.. code-block:: bash

    pip install f3dasm_optimize

More information about this extension can be found in the `f3dasm_optimize documentation page <https://f3dasm-optimize.readthedocs.io/en/latest/>`_