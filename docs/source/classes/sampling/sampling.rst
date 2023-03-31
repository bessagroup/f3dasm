Sampling
========

Creating a sampler
------------------

A new sampler can be created by initializing the sampler with:

* A :class:`~f3dasm.design.design.DesignSpace`
* A random :attr:`~f3dasm.sampling.sampler.Sampler.seed` (optional)


.. code-block:: python

  ran = f3dasm.sampling.RandomUniformSampling(design=design, seed=42)
  
Then we can evoke sampling by calling the :meth:`~f3dasm.sampling.sampler.Sampler.get_samples` method:

.. code-block:: python

  N = 100 # Number of samples
  data_ran = ran.get_samples(numsamples=N)
  
This will return a :class:`~f3dasm.design.experimentdata.ExperimentData` object filled with the requested samples.

.. _implemented samplers:

Implemented samplers
--------------------

The following implementations of samplers can found under the :mod:`f3dasm.sampling` module: 

======================== ====================================================================== ===========================================================================================================
Name                      Docs of the Python class                                              Reference
======================== ====================================================================== ===========================================================================================================
Random Uniform sampling  :class:`f3dasm.sampling.randomuniform.RandomUniform`                   `numpy.random.uniform <https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html>`_
Latin Hypercube sampling :class:`f3dasm.sampling.latinhypercube.LatinHypercube`                 `SALib.latin <https://salib.readthedocs.io/en/latest/api/SALib.sample.html?highlight=latin%20hypercube#SALib.sample.latin.sample>`_
Sobol Sequence sampling  :class:`f3dasm.sampling.sobolsequence.SobolSequence`                   `SALib.sobol_sequence <https://salib.readthedocs.io/en/latest/api/SALib.sample.html?highlight=sobol%20sequence#SALib.sample.sobol_sequence.sample>`_
======================== ====================================================================== ===========================================================================================================

Implement your own sampler
--------------------------

Implementing a new sampler goes as follows

* We create a new class inhereting from the :class:`~f3dasm.sampling.sampler.Sampler` class
* We have to implement our own :func:`~f3dasm.sampling.sampler.Sampler.sample_continuous` function:

.. note::

   We can also implement sampling strategies for all the other parameters but this is not necessary

This :func:`~f3dasm.sampling.sampler.Sampler.sample_continuous` function inputs the number of samples you want to create and returns a 2D numpy-array with the coordinates of those samples

.. code-block:: python

   class NewSampler(f3dasm.Sampler):
      def sample_continuous(self, numsamples: int) -> np.ndarray:
         ...

