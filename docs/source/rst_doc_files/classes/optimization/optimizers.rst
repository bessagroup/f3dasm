.. _optimization:

Optimizer
=========

The :class:`~f3dasm.optimization.Optimizer` class is used to find the minimum of a particular quantity of interest through an iterative fashion.

* The :meth:`~f3dasm.optimization.Optimizer.update_step` method takes a :class:`~f3dasm.datageneration.DataGenerator` object and outputs a tuple containing the position and evaluated value of the next iteration.
* The :meth:`~f3dasm.experimentdata.ExperimentData.optimize` method from :class:`~f3dasm.experimentdata.ExperimentData` is used to start the optimization process. It takes the number of iterations, the optimizer object and a :class:`~f3dasm.datageneration.DataGenerator` object as arguments. For every iteration, the :meth:`~f3dasm.optimization.Optimizer.update_step` method is called and the results are stored in the :class:`~f3dasm.design.ExperimentData` object.


.. image:: ../../../img/f3dasm-optimizer.png
    :width: 70%
    :align: center

|

Create an optimizer
-------------------

First, we have to determine the suitable search-space by creating a :class:`~f3dasm.design.Domain` object.

.. code-block:: python

    from f3dasm import Domain, ContinuousParameter

    domain = Domain(input_space={'x0': ContinuousParameter(lower_bound=0.0, upper_bound=1.0), 
                                    'x1': ContinuousParameter(lower_bound=0.0, upper_bound=1.0)})


Next, we have to create initial samples. We can use the  Latin-hypercube sampler to create samples:

.. code-block:: python

    from f3dasm.sampling import LatinHypercube

    data.from_sampling(sampler='latin', domain=domain, n_samples=10, seed=42)

We will use the ``"L-BFGS-B"`` optimizer to find the minimum. For built-in optimizer we can use the name of the optimizer:

.. code-block:: python

    data.optimize(optimizer='L-BFGS-B', iterations=100, data_generator='ackley')

.. note::

    You can pass hyperparameters of the optimizer as a dictionary to the ``optimize()`` method


Implement your own optimizer
----------------------------

To implement your own optimizer, you have to create a class that inherits from the :class:`~f3dasm.optimization.Optimizer` class 
and implement the :meth:`~f3dasm.optimization.Optimizer.update_step` method. 
The :meth:`~f3dasm.optimization.Optimizer.update_step` method takes a :class:`~f3dasm.datageneration.DataGenerator` object and outputs a tuple containing the position and evaluated value of the next iteration.

.. code-block:: python

    from f3dasm import Optimizer, DataGenerator

    class MyOptimizer(Optimizer):
        def update_step(self, data_generator: DataGenerator):
            # calculate the next position according to your update strategy
            return x_new, y_new


You can access the history of evaluations through the ``self.data`` attribute. This contains a copy of the ``ExperimentData`` object.

.. _implemented optimizers:

Implemented optimizers
----------------------

The following implementations of optimizers can found under the :mod:`f3dasm.optimization` module: 
These are ported from `scipy-optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_

======================== ========================================================================= ===============================================================================================
Name                     Key-word                                                                  Reference
======================== ========================================================================= ===============================================================================================
Conjugate Gradient       ``"CG"``                                                                  `scipy.minimize CG <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cg.html>`_
L-BFGS-B                 ``"LBFGSB"``                                                              `scipy.minimize L-BFGS-B <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_
Nelder Mead              ``"NelderMead"``                                                          `scipy.minimize NelderMead <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html>`_
Random search            ``"RandomSearch"``                                                        `numpy <https://numpy.org/doc/>`_
======================== ========================================================================= ===============================================================================================
