Built-in functionalities
========================

.. _implemented samplers:

Implemented samplers
--------------------

The following built-in implementations of samplers can be used in the data-driven process.

======================== ====================================================================== ===========================================================================================================
Name                     Method                                                                 Reference
======================== ====================================================================== ===========================================================================================================
``"random"``             Random Uniform sampling                                                `numpy.random.uniform <https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html>`_
``"latin"``              Latin Hypercube sampling                                               `SALib.latin <https://salib.readthedocs.io/en/latest/api/SALib.sample.html?highlight=latin%20hypercube#SALib.sample.latin.sample>`_
``"sobol"``              Sobol Sequence sampling                                                `SALib.sobol_sequence <https://salib.readthedocs.io/en/latest/api/SALib.sample.html?highlight=sobol%20sequence#SALib.sample.sobol_sequence.sample>`_
``"grid"``               Grid Search sampling                                                   `itertools.product <https://docs.python.org/3/library/itertools.html#itertools.product>`_
======================== ====================================================================== ===========================================================================================================

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

======================== ====================================================== ===========================
Name                      Docs of the Python class                              Data-generator argument
======================== ====================================================== ===========================
Ackley N. 2              :class:`~f3dasm.datageneration.AckleyN2`               ``"Ackley N. 2"``
Bohachevsky N. 1         :class:`~f3dasm.datageneration.BohachevskyN1`          ``"Bohachevsky N. 1"``
Booth                    :class:`~f3dasm.datageneration.Booth`                  ``"Booth"``
Brent                    :class:`~f3dasm.datageneration.Brent`                  ``"Brent"``
Brown                    :class:`~f3dasm.datageneration.Brown`                  ``"Brown"``
Bukin N. 6               :class:`~f3dasm.datageneration.BukinN6`                ``"Bukin N. 6"``
Dixon Price              :class:`~f3dasm.datageneration.DixonPrice`             ``"Dixon Price"``
Exponential              :class:`~f3dasm.datageneration.Exponential`            ``"Exponential"``
Matyas                   :class:`~f3dasm.datageneration.Matyas`                 ``"Matyas"``
McCormick                :class:`~f3dasm.datageneration.McCormick`              ``"McCormick"``
Perm 0, d, beta          :class:`~f3dasm.datageneration.PermZeroDBeta`          ``"Perm 0, d, beta"``
Powell                   :class:`~f3dasm.datageneration.Powell`                 ``"Powell"``
Rotated Hyper-Ellipsoid  :class:`~f3dasm.datageneration.RotatedHyperEllipsoid`  ``"Rotated Hyper-Ellipsoid"``
Schwefel 2.20            :class:`~f3dasm.datageneration.Schwefel2_20`           ``"Schwefel 2.20"``
Schwefel 2.21            :class:`~f3dasm.datageneration.Schwefel2_21`           ``"Schwefel 2.21"``
Schwefel 2.22            :class:`~f3dasm.datageneration.Schwefel2_22`           ``"Schwefel 2.22"``
Schwefel 2.23            :class:`~f3dasm.datageneration.Schwefel2_23`           ``"Schwefel 2.23"``
Sphere                   :class:`~f3dasm.datageneration.Sphere`                 ``"Sphere"``
Sum Squares              :class:`~f3dasm.datageneration.SumSquares`             ``"Sum Squares"``
Thevenot                 :class:`~f3dasm.datageneration.Thevenot`               ``"Thevenot"``
Trid                     :class:`~f3dasm.datageneration.Trid`                   ``"Trid"``
Xin She Yang N.3         :class:`~f3dasm.datageneration.XinSheYangN3`           ``"Xin She Yang N.3"``
Xin-She Yang N.4         :class:`~f3dasm.datageneration.XinSheYangN4`           ``"Xin-She Yang N.4"``
======================== ====================================================== ===========================



Seperable functions
^^^^^^^^^^^^^^^^^^^

======================== ============================================== ============================
Name                     Docs of the Python class                       Data-generator argument
======================== ============================================== ============================
Ackley                   :class:`~f3dasm.datageneration.Ackley`         ``"Ackley"``
Bohachevsky N. 1         :class:`~f3dasm.datageneration.BohachevskyN1`  ``"Bohachevsky N. 1"``
Easom                    :class:`~f3dasm.datageneration.Easom`          ``"Easom"``
Egg Crate                :class:`~f3dasm.datageneration.EggCrate`       ``"Egg Crate"``
Exponential              :class:`~f3dasm.datageneration.Exponential`    ``"Exponential"``
Griewank                 :class:`~f3dasm.datageneration.Griewank`       ``"Griewank"``
Michalewicz              :class:`~f3dasm.datageneration.Michalewicz`    ``"Michalewicz"``
Powell                   :class:`~f3dasm.datageneration.Powell`         ``"Powell"``
Qing                     :class:`~f3dasm.datageneration.Qing`           ``"Qing"``
Quartic                  :class:`~f3dasm.datageneration.Quartic`        ``"Quartic"``
Rastrigin                :class:`~f3dasm.datageneration.Rastrigin`      ``"Rastrigin"``
Schwefel                 :class:`~f3dasm.datageneration.Schwefel`       ``"Schwefel"``
Schwefel 2.20            :class:`~f3dasm.datageneration.Schwefel2_20`   ``"Schwefel 2.20"``
Schwefel 2.21            :class:`~f3dasm.datageneration.Schwefel2_21`   ``"Schwefel 2.21"``
Schwefel 2.22            :class:`~f3dasm.datageneration.Schwefel2_22`   ``"Schwefel 2.22"``
Schwefel 2.23            :class:`~f3dasm.datageneration.Schwefel2_23`   ``"Schwefel 2.23"``
Sphere                   :class:`~f3dasm.datageneration.Sphere`         ``"Sphere"``
Styblinski Tank          :class:`~f3dasm.datageneration.StyblinskiTank` ``"Styblinski Tank"``
Sum Squares              :class:`~f3dasm.datageneration.SumSquares`     ``"Sum Squares"``
Thevenot                 :class:`~f3dasm.datageneration.Thevenot`       ``"Thevenot"``
Xin She Yang             :class:`~f3dasm.datageneration.XinSheYang`     ``"Xin She Yang"``
======================== ============================================== ============================

Multimodal functions
^^^^^^^^^^^^^^^^^^^^

======================== ================================================ ==========================
Name                     Docs of the Python class                         Data-generator argument
======================== ================================================ ==========================
Ackley                   :class:`~f3dasm.datageneration.Ackley`           ``"Ackley"``
Ackley N. 3              :class:`~f3dasm.datageneration.AckleyN3`         ``"Ackley N. 3"``
Ackley N. 4              :class:`~f3dasm.datageneration.AckleyN4`         ``"Ackley N. 4"``
Adjiman                  :class:`~f3dasm.datageneration.Adjiman`          ``"Adjiman"``
Bartels                  :class:`~f3dasm.datageneration.Bartels`          ``"Bartels"``
Beale                    :class:`~f3dasm.datageneration.Beale`            ``"Beale"``
Bird                     :class:`~f3dasm.datageneration.Bird`             ``"Bird"``
Bohachevsky N. 2         :class:`~f3dasm.datageneration.BohachevskyN2`    ``"Bohachevsky N. 2"``
Bohachevsky N. 3         :class:`~f3dasm.datageneration.BohachevskyN3`    ``"Bohachevsky N. 3"``
Branin                   :class:`~f3dasm.datageneration.Branin`           ``"Branin"``
Bukin N. 6               :class:`~f3dasm.datageneration.BukinN6`          ``"Bukin N. 6"``
Colville                 :class:`~f3dasm.datageneration.Colville`         ``"Colville"``
Cross-in-Tray            :class:`~f3dasm.datageneration.CrossInTray`      ``"Cross-in-Tray"``
De Jong N. 5             :class:`~f3dasm.datageneration.DeJongN5`         ``"De Jong N. 5"``
Deckkers-Aarts           :class:`~f3dasm.datageneration.DeckkersAarts`    ``"Deckkers-Aarts"``
Easom                    :class:`~f3dasm.datageneration.Easom`            ``"Easom"``
Egg Crate                :class:`~f3dasm.datageneration.EggCrate`         ``"Egg Crate"``
Egg Holder               :class:`~f3dasm.datageneration.EggHolder`        ``"Egg Holder"``
Goldstein-Price          :class:`~f3dasm.datageneration.GoldsteinPrice`   ``"Goldstein-Price"``
Happy Cat                :class:`~f3dasm.datageneration.HappyCat`         ``"Happy Cat"``
Himmelblau               :class:`~f3dasm.datageneration.Himmelblau`       ``"Himmelblau"``
Holder-Table             :class:`~f3dasm.datageneration.HolderTable`      ``"Holder-Table"``
Keane                    :class:`~f3dasm.datageneration.Keane`            ``"Keane"``
Langermann               :class:`~f3dasm.datageneration.Langermann`       ``"Langermann"``
Levy                     :class:`~f3dasm.datageneration.Levy`             ``"Levy"``
Levy N. 13               :class:`~f3dasm.datageneration.LevyN13`          ``"Levy N. 13"``
McCormick                :class:`~f3dasm.datageneration.McCormick`        ``"McCormick"``
Michalewicz              :class:`~f3dasm.datageneration.Michalewicz`      ``"Michalewicz"``
Periodic                 :class:`~f3dasm.datageneration.Periodic`         ``"Periodic"``
Perm d, beta             :class:`~f3dasm.datageneration.PermDBeta`        ``"Perm d, beta"``
Qing                     :class:`~f3dasm.datageneration.Qing`             ``"Qing"``
Quartic                  :class:`~f3dasm.datageneration.Quartic`          ``"Quartic"``
Rastrigin                :class:`~f3dasm.datageneration.Rastrigin`        ``"Rastrigin"``
Rosenbrock               :class:`~f3dasm.datageneration.Rosenbrock`       ``"Rosenbrock"``
Salomon                  :class:`~f3dasm.datageneration.Salomon`          ``"Salomon"``
Schwefel                 :class:`~f3dasm.datageneration.Schwefel`         ``"Schwefel"``
Shekel                   :class:`~f3dasm.datageneration.Shekel`           ``"Shekel"``
Shubert                  :class:`~f3dasm.datageneration.Shubert`          ``"Shubert"``
Shubert N. 3             :class:`~f3dasm.datageneration.ShubertN3`        ``"Shubert N. 3"``
Shubert N. 4             :class:`~f3dasm.datageneration.ShubertN4`        ``"Shubert N. 4"``
Styblinski Tank          :class:`~f3dasm.datageneration.StyblinskiTank`   ``"Styblinski Tank"``
Thevenot                 :class:`~f3dasm.datageneration.Thevenot`         ``"Thevenot"``
Xin She Yang             :class:`~f3dasm.datageneration.XinSheYang`       ``"Xin She Yang"``
Xin She Yang N.2         :class:`~f3dasm.datageneration.XinSheYangN2`     ``"Xin She Yang N.2"``
======================== ================================================ ==========================

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