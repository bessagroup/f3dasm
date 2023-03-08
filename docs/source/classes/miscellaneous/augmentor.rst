Function Augmentor
==================

Usage
-----

In order to further diversify your benchmark functions, it is possible to add add data augmentation to you benchmark functions.
Within :code:`f3dasm` this is done with the :class:`~f3dasm.functions.adapters.augmentor.Augmentor` class.
The following three augmentation operations are supported in :code:`f3dasm`:

- :class:`~f3dasm.functions.adapters.augmentor.Scale`: Scaling the boundaries of the function to another set of lower and upper boundaries
- :class:`~f3dasm.functions.adapters.augmentor.Offset`: Offsetting the benchmarkfunction by a constant vector
- :class:`~f3dasm.functions.adapters.augmentor.Noise`: Adding Gaussian noise to the objective value.

You can create any combination of augmentors and supply them in lists to create a :class:`~f3dasm.functions.adapters.augmentor.FunctionAugmentor` object.

- You can add a list of augmentors that work on the **input vector** to the :attr:`~f3dasm.functions.adapters.augmentor.FunctionAugmentor.input_augmentors` attribute with the :meth:`~f3dasm.functions.adapters.augmentor.FunctionAugmentor.add_input_augmentor` method.
- You can add a list of augmentors that work on the **objective value** to the :attr:`~f3dasm.functions.adapters.augmentor.FunctionAugmentor.output_augmentors` attribute with the :meth:`~f3dasm.functions.adapters.augmentor.FunctionAugmentor.add_output_augmentor` method.

Whenever you evaluate the benchmark function, the input and output vectors will be manipulated by the augmentors in the :class:`~f3dasm.functions.adapters.augmentor.FunctionAugmentor` in order.
You can retrieve the original value from a vector that has been manipulated by the augmentors by calling the :meth:`~f3dasm.functions.adapters.augmentor.FunctionAugmentor.augment_reverse_input` method.

When a benchmarkfunction object is created, an empty :class:`~f3dasm.functions.adapters.augmentor.FunctionAugmentor` is created and stored as attribute (:class:`~f3dasm.functions.Function.augmentor`). 
If you provide one of the following initialization attributes to the object, augmentors are created and added accordingly:

- :attr:`~f3dasm.functions.adapters.pybenchfunction.PyBenchFunction.scale_bounds`, if set not to None
- :attr:`~f3dasm.functions.adapters.pybenchfunction.PyBenchFunction.offset` if set to True, (default value is True)
- :attr:`~f3dasm.functions.adapters.pybenchfunction.PyBenchFunction.noise` if set not to None

Create your own augmentor
-------------------------

In order to create your own augmentor, create a new class and inheret from the base :class:`~f3dasm.functions.adapters.augmentor.Augmentor` class:

.. code-block:: python

  class NewAugmentor(Augmentor):
      """
      Base class for operations that augment an loss-funciton
      """
  
      def augment(self, input: np.ndarray) -> np.ndarray:
          ...
  
      def reverse_augment(self, output: np.ndarray) -> np.ndarray:
          ...




