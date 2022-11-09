Sampling
========

Usage
-----

A new sampler can be created by initializing the sampler with:

* A :class:`~f3dasm.base.design.DesignSpace`
* A random :attr:`~f3dasm.base.samplingmethod.SamplingInterface.seed` (optional)


.. code-block:: python

  ran = f3dasm.sampling.RandomUniformSampling(design=design, seed=42)
  
Then we can evoke sampling by calling the :meth:`~f3dasm.base.samplingmethod.SamplingInterface.get_samples` method:

.. code-block:: python

  N = 100 # Number of samples
  data_ran = ran.get_samples(numsamples=N)
  
This will return a :class:`~f3dasm.base.data.Data` object filled with the requested samples.


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

API Documentation
-----------------

Latin Hypercube sampling
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: f3dasm.sampling.latinhypercube
   :members:
   :undoc-members:
   :show-inheritance:

Random Uniform sampling
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: f3dasm.sampling.randomuniform
   :members:
   :undoc-members:
   :show-inheritance:

Sobol sequence sampling
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: f3dasm.sampling.sobolsequence
   :members:
   :undoc-members:
   :show-inheritance: