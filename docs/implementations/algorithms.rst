Optimizers
==========

Usage
-----

We will use the CMAES optimizer to find the minimum. We can find an implementation in the :mod:`f3dasm.optimization` module:

.. code-block:: python

    optimizer = f3dasm.optimization.CMAES(data=samples_lhs)

By calling the :func:`~f3dasm.optimization.Optimizer.iterate` method and specifying the function and the number of iterations, we will start the optimization process:

.. code-block:: python

    optimizer.iterate(iterations=100, function=f)

After that, we can extract the data:

.. code-block:: python

    cmaes_data = optimizer.extract_data()



Implemented optimizers
----------------------

The following implementations of optimizers can found under the :mod:`f3dasm.optimization` module: 

These are ported from several libraries such as `GPyOpt <https://sheffieldml.github.io/GPyOpt/>`_, `scipy-optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_, `tensorflow <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_ and `pygmo <https://esa.github.io/pygmo2/>`_.


Pygmo implementations
^^^^^^^^^^^^^^^^^^^^^

These derivative-free global optimizers are ported from the `pygmo <https://esa.github.io/pygmo2/>`_ Python library: 

======================== ========================================================================== =======================================================================================================
Name                      Docs of the Python class                                                  Reference
======================== ========================================================================== =======================================================================================================
CMAES                    :class:`~f3dasm.optimization.cmaes.CMAES`                                  `pygmo cmaes <https://esa.github.io/pygmo2/algorithms.html#pygmo.cmaes>`_
PSO                      :class:`~f3dasm.optimization.pso.PSO`                                      `pygmo pso_gen <https://esa.github.io/pygmo2/algorithms.html#pygmo.pso_gen>`_
SGA                      :class:`~f3dasm.optimization.sga.SGA`                                      `pygmo sga <https://esa.github.io/pygmo2/algorithms.html#pygmo.sga>`_
SEA                      :class:`~f3dasm.optimization.sea.SEA`                                      `pygmo sea <https://esa.github.io/pygmo2/algorithms.html#pygmo.sea>`_
XNES                     :class:`~f3dasm.optimization.xnes.XNES`                                    `pygmo xnes <https://esa.github.io/pygmo2/algorithms.html#pygmo.xnes>`_
Differential Evolution   :class:`~f3dasm.optimization.differentialevolution.DifferentialEvolution`  `pygmo de <https://esa.github.io/pygmo2/algorithms.html#pygmo.de>`_
Simulated Annealing      :class:`~f3dasm.optimization.simulatedannealing.SimulatedAnnealing`        `pygmo simulated_annealing <https://esa.github.io/pygmo2/algorithms.html#pygmo.simulated_annealing>`_
======================== ========================================================================== =======================================================================================================

Scipy Implementations
^^^^^^^^^^^^^^^^^^^^^

These optimizers are ported from the `scipy <https://scipy.org/>`_ Python library: 

======================== ========================================================================= ===============================================================================================
Name                      Docs of the Python class                                                 Reference
======================== ========================================================================= ===============================================================================================
CG                       :class:`~f3dasm.optimization.cg.CG`                                        `scipy.minimize CG <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cg.html>`_
LBFGSB                   :class:`~f3dasm.optimization.lbfgsb.LBFGSB`                                `scipy.minimize L-BFGS-B <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_
NelderMead               :class:`~f3dasm.optimization.neldermead.NelderMead`                        `scipy.minimize NelderMead <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html>`_
NelderMead               :class:`~f3dasm.optimization.cobyla.COBYLA`                                `scipy.minimize COBYLA <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html>`_

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

API Documentation
-----------------

Adam
^^^^

.. automodule:: f3dasm.optimization.adam
   :members:
   :show-inheritance:

Adamax
^^^^^^

.. automodule:: f3dasm.optimization.adamax
   :members:
   :show-inheritance:

Bayesian Optimization
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: f3dasm.optimization.bayesianoptimization
   :members:
   :show-inheritance:

Bayesian Optimization Torch
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: f3dasm.optimization.bayesianoptimization_torch
   :members:
   :show-inheritance:

Conjugate Gradient
^^^^^^^^^^^^^^^^^^

.. automodule:: f3dasm.optimization.cg
   :members:
   :show-inheritance:

CMAES
^^^^^

.. automodule:: f3dasm.optimization.cmaes
   :members:
   :show-inheritance:

COBYLA
^^^^^^

.. automodule:: f3dasm.optimization.cobyla
   :members:
   :show-inheritance:

Differential Evolution
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: f3dasm.optimization.differentialevolution
   :members:
   :show-inheritance:

FTRL
^^^^

.. automodule:: f3dasm.optimization.ftrl
   :members:
   :show-inheritance:

LBFGSB
^^^^^^

.. automodule:: f3dasm.optimization.lbfgsb
   :members:
   :show-inheritance:

Nadam
^^^^^

.. automodule:: f3dasm.optimization.nadam
   :members:
   :show-inheritance:

Nelder Mead
^^^^^^^^^^^

.. automodule:: f3dasm.optimization.neldermead
   :members:
   :show-inheritance:

PSO
^^^

.. automodule:: f3dasm.optimization.pso
   :members:
   :show-inheritance:

Random Search
^^^^^^^^^^^^^

.. automodule:: f3dasm.optimization.randomsearch
   :members:
   :show-inheritance:

RMSprop
^^^^^^^

.. automodule:: f3dasm.optimization.rmsprop
   :members:
   :show-inheritance:

SADE
^^^^

.. automodule:: f3dasm.optimization.sade
   :members:
   :show-inheritance:

SEA
^^^

.. automodule:: f3dasm.optimization.sea
   :members:
   :show-inheritance:

SGA
^^^

.. automodule:: f3dasm.optimization.sga
   :members:
   :show-inheritance:

SGD
^^^

.. automodule:: f3dasm.optimization.sgd
   :members:
   :show-inheritance:

Simulated Annealing
^^^^^^^^^^^^^^^^^^^

.. automodule:: f3dasm.optimization.simulatedannealing
   :members:
   :show-inheritance:

XNES
^^^^

.. automodule:: f3dasm.optimization.xnes
   :members:
   :show-inheritance: