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
CMAES                    :class:`~f3dasm.optimization.cmaes.CMAES`                               `pygmo cmaes <https://esa.github.io/pygmo2/algorithms.html#pygmo.cmaes>`_
PSO                      :class:`~f3dasm.optimization.pso.PSO`                                   `pygmo pso_gen <https://esa.github.io/pygmo2/algorithms.html#pygmo.pso_gen>`_
SGA                      :class:`~f3dasm.optimization.sga.SGA`                                   `pygmo sga <https://esa.github.io/pygmo2/algorithms.html#pygmo.sga>`_
XNES                     :class:`~f3dasm.optimization.xnes.XNES`                                 `pygmo xnes <https://esa.github.io/pygmo2/algorithms.html#pygmo.xnes>`_
======================== ====================================================================== ===============================================================================

Scipy Implementations
^^^^^^^^^^^^^^^^^^^^^

These optimizers are ported from the `scipy <https://scipy.org/>`_ Python library: 

======================== ========================================================================= ===============================================================================================
Name                      Docs of the Python class                                                 Reference
======================== ========================================================================= ===============================================================================================
CG                       :class:`~f3dasm.optimization.cg.CG`                                        `scipy.minimize CG <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cg.html>`_
DifferentialEvolution    :class:`~f3dasm.optimization.differentialevolution.DifferentialEvolution`  `scipy.optimize Differential Evolution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution>`_
DualAnnealing            :class:`~f3dasm.optimization.dualannealing.DualAnnealing`                  `scipy.optimize Dual Annealing <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html#scipy.optimize.dual_annealing>`_
LBFGSB                   :class:`~f3dasm.optimization.lbfgsb.LBFGSB`                                `scipy.minimize L-BFGS-B <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_
NelderMead               :class:`~f3dasm.optimization.neldermead.NelderMead`                        `scipy.minimize NelderMead <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html>`_
======================== ========================================================================= ===============================================================================================


GPyOpt Implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

======================== ========================================================================= ======================================================
Name                      Docs of the Python class                                                 Reference
======================== ========================================================================= ======================================================
Bayesian Optimization    :class:`~f3dasm.optimization.bayesianoptimization.BayesianOptimization`    `GPyOpt <https://gpyopt.readthedocs.io/en/latest/>`_
======================== ========================================================================= ======================================================

Tensorflow Keras optimizers
^^^^^^^^^^^^^^^^^^^^^^^^^^^

======================== ====================================================================== =====================================================================================================
Name                      Docs of the Python class                                              Reference
======================== ====================================================================== =====================================================================================================
SGD                      :class:`~f3dasm.optimization.sgd.SGD`                                   `tf.keras.optimizers.SGD <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD>`_
RMSprop                  :class:`~f3dasm.optimization.rmsprop.RMSprop`                           `tf.keras.optimizers.RMSprop <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop>`_
Adam                     :class:`~f3dasm.optimization.adam.Adam`                                 `tf.keras.optimizers.Adam <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam>`_
Nadam                    :class:`~f3dasm.optimization.nadam.Nadam`                               `tf.keras.optimizers.Nadam <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Nadam>`_
Adamax                   :class:`~f3dasm.optimization.adamax.Adamax`                             `tf.keras.optimizers.Adamax <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adamax>`_
Ftrl                     :class:`~f3dasm.optimization.ftrl.Ftrl`                                 `tf.keras.optimizers.Ftrl <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Ftrl>`_
======================== ====================================================================== =====================================================================================================

Self implemented optimizers
^^^^^^^^^^^^^^^^^^^^^^^^^^^

======================== ====================================================================== ==================
Name                      Docs of the Python class                                              Reference
======================== ====================================================================== ==================
RandomSearch             :class:`~f3dasm.optimization.randomsearch.RandomSearch`                 self implemented
======================== ====================================================================== ==================
