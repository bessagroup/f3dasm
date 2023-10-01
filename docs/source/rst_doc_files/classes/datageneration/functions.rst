.. _benchmark-functions:

Benchmark functions
===================

:mod:`f3dasm` comes with a set of benchmark functions that can be used to test the performance of 
optimization algorithms or to mock some expensive simulation in order to test the workflow


.. note::

  The gradients of the benchmark functions are computed with the automated differentiation package `autograd <https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html>`_.


In order to augment the benchmark functions, you can provide 4 keyword arguments to the :class:`~f3dasm.design.ExperimentData.evaluate` method:

* ``scale_bounds``: A 2D list of float that define the scaling lower and upper boundaries for each dimension. The normal benchmark function box-constraints will be scaled to these boundaries.
* ``noise``: A float that defines the standard deviation of the Gaussian noise that is added to the objective value.
* ``offset``: A boolean value. If ``True``, the benchmark function will be offset by a constant vector that will be randomly generated.
* ``seed``: Seed for the random number generator for the ``noise`` and ``offset`` calculations.

Benchmark functions can substitute the expensive simulation in the 
:class:`~f3dasm.design.ExperimentData` object by providing the name of the function as the ``data_generator`` argument:

.. code-block:: python

   from f3dasm.design import ExperimentData, Domain
   import numpy as np

   domain = Domain(...)
   # Create the ExperimentData object
   experiment_data = ExperimentData.from_sampling('random', domain=domain, n_samples=10, seed=42)

    # Evaluate the Ackley function with noise, offset and scale_bounds
   experiment_data.evaluate('Ackley', kwargs={'noise': 0.1, 'scale_bounds': [[0., 1.], [-1., 1.]], 'offset': True, 'seed': 42})


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
