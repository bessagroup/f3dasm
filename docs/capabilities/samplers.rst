Samplers
========

Usage
-----

We have to give initialize each sampler with a :class:`~f3dasm.base.design.DesignSpace`:

.. code-block:: python

  ran = f3dasm.sampling.RandomUniformSampling(doe=design)
  
Then we can evoke sampling by calling the :meth:`~f3dasm.base.samplingmethod.SamplingInterface.get_samples` method:

.. code-block:: python

  N = 100 # Number of samples
  data_ran = ran.get_samples(numsamples=N)
  
This will return a :class:`~f3dasm.base.data.Data` object filled with the requested samples.

Implement your own sampler
--------------------------


Implemented samplers
--------------------

The following implementations of samplers can found under the :mod:`f3dasm.sampling` module: 

======================== ====================================================================== ===========================================================================================================
Name                      Docs of the Python class                                              Reference
======================== ====================================================================== ===========================================================================================================
Random Uniform sampling  :class:`f3dasm.sampling.samplers.RandomUniformSampling`                `numpy.random.uniform <https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html>`_
Latin Hypercube sampling :class:`f3dasm.sampling.samplers.LatinHypercubeSampling`               `SALib.latin <https://salib.readthedocs.io/en/latest/api/SALib.sample.html?highlight=latin%20hypercube#SALib.sample.latin.sample>`_
Sobol Sequence sampling  :class:`f3dasm.sampling.samplers.SobolSequenceSampling`                `SALib.sobol_sequence <https://salib.readthedocs.io/en/latest/api/SALib.sample.html?highlight=sobol%20sequence#SALib.sample.sobol_sequence.sample>`_

======================== ====================================================================== ===========================================================================================================
