Implemented optimizers
======================

Creat an optimizer
------------------

We will use the CMAES optimizer to find the minimum. We can find an implementation in the :mod:`f3dasm.optimization` module:

.. code-block:: python

    optimizer = f3dasm.optimization.CMAES(data=samples_lhs)

By calling the :func:`~f3dasm.optimization.optimizer.Optimizer.iterate` method and specifying the function and the number of iterations, we will start the optimization process:

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
COBYLA                   :class:`~f3dasm.optimization.cobyla.COBYLA`                                `scipy.minimize COBYLA <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html>`_

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

Implement your own optimizer
----------------------------

First, we create a class storing the potential hyper-parameters for our optimizers. Even if we our optimizer doesn't have hyper-parameters, you still have to create class

This class has to be inhereted from the :class:`~f3dasm.optimization.optimizer.OptimizerParameters` class. This inhereted class consists two mandatory attributes: 

* :attr:`~f3dasm.optimization.optimizer.OptimizerParameters.population`: how many points are created for each update step. Defaults to 1
* :attr:`~f3dasm.optimization.optimizer.OptimizerParameters.force_bounds`: if the optimizer is forced to stay between the design bounds. Defaults to True. Currently does not work when set to False!

.. code-block:: python

    @dataclass
    class NewOptimizer_Parameters(f3dasm.OptimizerParameters):
    """Example of hyperparameters"""

    example_hyperparameter_1: float = 0.999
    example_hyperparameter_2: bool = True


Next, we create an new optimizer by inheriting from the :class:`~f3dasm.optimization.optimizer.Optimizer` class

* We create a class attribute :attr:`~f3dasm.optimization.optimizer.Optimizer.parameter` and initialize it without any arguments in order to use the defaults specified above
* The only function we have to implement is the :func:`~f3dasm.optimization.optimizer.Optimizer.update_step` function, which takes a :class:`~f3dasm.base.function.Function` and outputs a tuple containing the position and evaluated value of the next iteration
* The :func:`~f3dasm.optimization.optimizer.Optimizer.init_parameters` function is optional. It can be used to store dynamic hyper-parameters that update throughout updating


.. code-block:: python

    class NewOptimizer(f3dasm.Optimizer):
    """Example of implement your own optimizer"""

    parameter: NewOptimizer_Parameters = NewOptimizer_Parameters()

    def init_parameters(self):
        """Set the dynamic initialization parameters. These are resetted every time the iterate method is called."""
        pass

    def update_step(self, function: f3dasm.Function) -> Tuple[np.ndarray, np.ndarray]:
        """Custom update step for your own optimizer

        Parameters
        ----------
        function
            objective function that is being optimized

        Returns
        -------
            tuple of resulting input and output parameter
        """
        return x, y

In order to use the optimizer, we call the :func:`~f3dasm.optimization.optimizer.Optimizer.iterate` method, which for-loops over the :func:`~f3dasm.optimization.optimizer.Optimizer.update_step` method, appending the :code:`x` and :code:`y` values to the internal :attr:`~f3dasm.optimization.optimizer.Optimizer.data` attribute.


