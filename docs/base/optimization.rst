Optimizer interface
===================

Implement your own optimizer
----------------------------
First we create new class denoting the hyper-parameters of the new optimizer. 
We do that by inheriting from the :class:`~f3dasm.base.optimization.OptimizerParameters`

.. code-block:: python

   @dataclass
   class NewOptimizer_Parameters:
    """Example of hyperparameters"""

    hyperparameter_1: float = 0.999
    hyperparameter_2: bool = True


Then you can create a new optimizer class by inhereting from the :class:`~f3dasm.base.optimization.Optimizer` class

.. code-block:: python

   class NewOptimizer(Optimizer):
    """Example of implement your own optimizer"""

    parameter: NewOptimizer_Parameters = NewOptimizer_Parameters()

    def update_step(self, function: Function) -> Tuple[np.ndarray, np.ndarray]:
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

API Documentation
-----------------

.. automodule:: f3dasm.base.optimization
   :members:
   :noindex:
   :show-inheritance: