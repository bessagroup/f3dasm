Sampling
========

Samplers take a :class:`~f3dasm.design.Domain` object and 
return a :class:`~f3dasm.design.ExperimentData` object filled with samples based on the sampling strategy.

.. _integrating-samplers:

Implement your sampling strategy
--------------------------------

Implementing your own sampling strategy in the data-driven process works as follows:

* Create a new class inhereting from the :class:`~f3dasm.sampling.Sampler` class
* Implement our own :func:`~f3dasm.sampling.Sampler.sample_continuous` function:

.. note::

   We can also implement sampling strategies for all the other parameters but this is not necessary

This :func:`~f3dasm.sampling.Sampler.sample_continuous` function inputs the number of samples you want to create and returns a 2D numpy-array with the coordinates of those samples

.. code-block:: python

   class NewSampler(f3dasm.Sampler):
      def sample_continuous(self, numsamples: int) -> np.ndarray:
         ...



Creating a sampler
------------------

Using the sampler
-----------------

A sampler object can be created by initializing the class with:

* A :class:`~f3dasm.design.Domain` object
* A random :attr:`~f3dasm.sampling.Sampler.seed` (optional)
* (Optionally) the number of samples :attr:`~f3dasm.samplingSampler.number_of_samples` to create


.. code-block:: python

  ran = f3dasm.sampling.RandomUniformSampling(design=design, seed=42)
  
Then we can evoke sampling by calling the :meth:`~f3dasm.sampling.Sampler.get_samples` method:

.. code-block:: python

  N = 100 # Number of samples
  data_ran = ran.get_samples(numsamples=N)
  
This will return a :class:`~f3dasm.design.ExperimentData` object filled with the requested samples.

.. _implemented samplers:

Implemented samplers
--------------------

The following implementations of samplers can found under the :mod:`f3dasm.sampling` module: 

.. ======================== ====================================================================== ===========================================================================================================
.. Name                      Docs of the Python class                                              Reference
.. ======================== ====================================================================== ===========================================================================================================
.. Random Uniform sampling  :class:`f3dasm.sampling.RandomUniform`                                 `numpy.random.uniform <https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html>`_
.. Latin Hypercube sampling :class:`f3dasm.sampling.LatinHypercube`                                `SALib.latin <https://salib.readthedocs.io/en/latest/api/SALib.sample.html?highlight=latin%20hypercube#SALib.sample.latin.sample>`_
.. Sobol Sequence sampling  :class:`f3dasm.sampling.SobolSequence`                                 `SALib.sobol_sequence <https://salib.readthedocs.io/en/latest/api/SALib.sample.html?highlight=sobol%20sequence#SALib.sample.sobol_sequence.sample>`_
.. ======================== ====================================================================== ===========================================================================================================

.. autosummary::
  :toctree: _autosummary

    f3dasm.sampling.RandomUniform
    f3dasm.sampling.LatinHypercube
    f3dasm.sampling.SobolSequence



