Sampling interface
==================

Implement your own sampler
--------------------------

Implementing a new sampler goes as follows

* We create a new class inhereting from the :class:`~f3dasm.base.samplingmethod.SamplingInterface` class
* We have to implement our own :func:`~f3dasm.base.samplingmethod.SamplingInterface.sample_continuous` function:

.. note::

   We can also implement sampling strategies for all the other parameters but this is not necessary

This :func:`~f3dasm.base.samplingmethod.SamplingInterface.sample_continuous` function inputs the number of samples you want to create and returns a 2D numpy-array with the coordinates of those samples

.. code-block:: python

   class NewSampler(f3dasm.SamplingInterface):
      def sample_continuous(self, numsamples: int) -> np.ndarray:
         ...


API Documentation
-----------------

.. automodule:: f3dasm.base.samplingmethod
   :members:
   :noindex:
   :show-inheritance: