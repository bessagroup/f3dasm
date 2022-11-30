Optimizer interface
===================

Implement your own optimizer
----------------------------
First, we create a class storing the potential hyper-parameters for our optimizers. Even if we our optimizer doesn't have hyper-parameters, you still have to create class

This class has to be inhereted from the :class:`~f3dasm.base.optimization.OptimizerParameters` class. This inhereted class consists two mandatory attributes: 

* :attr:`~f3dasm.base.optimization.OptimizerParameters.population`: how many points are created for each update step. Defaults to 1
* :attr:`~f3dasm.base.optimization.OptimizerParameters.force_bounds`: if the optimizer is forced to stay between the design bounds. Defaults to True. Currently does not work when set to False!

.. code-block:: python

    @dataclass
    class NewOptimizer_Parameters(f3dasm.OptimizerParameters):
    """Example of hyperparameters"""

    example_hyperparameter_1: float = 0.999
    example_hyperparameter_2: bool = True


Next, we create an new optimizer by inheriting from the :class:`~f3dasm.base.optimization.Optimizer` class

* We create a class attribute :attr:`~f3dasm.base.optimization.Optimizer.parameter` and initialize it without any arguments in order to use the defaults specified above
* The only function we have to implement is the :func:`~f3dasm.base.optimization.Optimizer.update_step` function, which takes a :class:`~f3dasm.base.function.Function` and outputs a tuple containing the position and evaluated value of the next iteration
* The :func:`~f3dasm.base.optimization.Optimizer.init_parameters` function is optional. It can be used to store dynamic hyper-parameters that update throughout updating


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

In order to use the optimizer, we call the :func:`~f3dasm.base.optimization.Optimizer.iterate` method, which for-loops over the :func:`~f3dasm.base.optimization.Optimizer.update_step` method, appending the :code:`x` and :code:`y` values to the internal :attr:`~f3dasm.base.optimization.Optimizer.data` attribute.

API Documentation
-----------------

.. automodule:: f3dasm.base.optimization
   :members:
   :noindex:
   :show-inheritance: