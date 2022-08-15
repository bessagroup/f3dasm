Optimizers
==========

Usage
-----



Implement your own optimizer
----------------------------


Implemented optimizers
----------------------

The following implementations of optimizers can found under the :mod:`f3dasm.optimization` module: 


Pygmo implementations
^^^^^^^^^^^^^^^^^^^^^

These derivative-free global optimizers are ported from the `pygmo <https://esa.github.io/pygmo2/>`_ Python library: 

======================== ====================================================================== ===============================================================================
Name                      Docs of the Python class                                              Reference
======================== ====================================================================== ===============================================================================
CMAES                    :class:`f3dasm.optimization.pygmo_implementations.CMAES`               `pygmo cmaes <https://esa.github.io/pygmo2/algorithms.html#pygmo.cmaes>`_
PSO                      :class:`f3dasm.optimization.pygmo_implementations.PSO`                 `pygmo pso_gen <https://esa.github.io/pygmo2/algorithms.html#pygmo.pso_gen>`_
SGA                      :class:`f3dasm.optimization.pygmo_implementations.SGA`                 `pygmo sga <https://esa.github.io/pygmo2/algorithms.html#pygmo.sga>`_
XNES                     :class:`f3dasm.optimization.pygmo_implementations.XNES`                `pygmo xnes <https://esa.github.io/pygmo2/algorithms.html#pygmo.xnes>`_
======================== ====================================================================== ===============================================================================

Scipy Implementations
^^^^^^^^^^^^^^^^^^^^^

These optimizers are ported from the `scipy <https://scipy.org/>`_ Python library: 

======================== ========================================================================= ===============================================================================================
Name                      Docs of the Python class                                                 Reference
======================== ========================================================================= ===============================================================================================
CG                       :class:`f3dasm.optimization.scipy_implementations.CG`                     `scipy.minimize CG <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cg.html>`_
DifferentialEvolution    :class:`f3dasm.optimization.scipy_implementations.DifferentialEvolution`  `scipy.optimize Differential Evolution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution>`_
DualAnnealing            :class:`f3dasm.optimization.scipy_implementations.DualAnnealing`          `scipy.optimize Dual Annealing <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html#scipy.optimize.dual_annealing>`_
LBFGSB                   :class:`f3dasm.optimization.scipy_implementations.LBFGSB`                 `scipy.minimize L-BFGS-B <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_
NelderMead               :class:`f3dasm.optimization.scipy_implementations.NelderMead`             `scipy.minimize NelderMead <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html>`_
======================== ========================================================================= ===============================================================================================

Self implemented optimizers
^^^^^^^^^^^^^^^^^^^^^^^^^^^

======================== ====================================================================== ==================
Name                      Docs of the Python class                                              Reference
======================== ====================================================================== ==================
Adam                     :class:`f3dasm.optimization.gradient_based_algorithms.Adam`            self implemented
Momentum                 :class:`f3dasm.optimization.gradient_based_algorithms.Momentum`        self implemented
SGD                      :class:`f3dasm.optimization.gradient_based_algorithms.SGD`             self implemented
RandomSearch             :class:`f3dasm.optimization.randomsearch.RandomSearch`                 self implemented
======================== ====================================================================== ==================
