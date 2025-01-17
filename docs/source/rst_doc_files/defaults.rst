Built-in functionalities
========================

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

More information about this extension can be found in the `f3dasm_optimize documentation page <https://bessagroup.github.io/f3dasm_optimize/>`_