Sampling
========

Samplers take a :class:`~f3dasm.design.domain.Domain` object and 
return a :class:`~f3dasm.design.experimentdata.ExperimentData` object filled with samples based on the sampling strategy.

Creating a sampler
------------------

Sampler from the default constructor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A new sampler can be created by initializing the sampler with:

* A :class:`~f3dasm.design.domain.Domain` object
* A random :attr:`~f3dasm.sampling.sampler.Sampler.seed` (optional)
* (Optionally) the number of samples :attr:`~f3dasm.sampling.sampler.Sampler.number_of_samples` to create


.. code-block:: python

  ran = f3dasm.sampling.RandomUniformSampling(design=design, seed=42)
  
Then we can evoke sampling by calling the :meth:`~f3dasm.sampling.sampler.Sampler.get_samples` method:

.. code-block:: python

  N = 100 # Number of samples
  data_ran = ran.get_samples(numsamples=N)
  
This will return a :class:`~f3dasm.design.experimentdata.ExperimentData` object filled with the requested samples.

.. _sampler-hydra:

Sampler from a hydra configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are using hydra to manage your configuration files, you can create a sampler from a configuration file.
Your configuration file should look like this:

.. code-block:: yaml
   :caption: config.yaml

   domain:
      input_space:
         param_1:
               _target_: f3dasm.ContinuousParameter
               lower_bound: -1.0
               upper_bound: 1.0
         param_2:
               _target_: f3dasm.DiscreteParameter
               lower_bound: 1
               upper_bound: 10
         param_3:
               _target_: f3dasm.CategoricalParameter
               categories: ['red', 'blue', 'green', 'yellow', 'purple']
         param_4:
               _target_: f3dasm.ConstantParameter
               value: some_value

   sampler:
      _target_: f3dasm.sampling.RandomUniform
      number_of_samples: 100
      seed: 42

You need to provide the :code:`sampler` and :code:`domain` keys of you config file to the :func:`~f3dasm.sampling.sampler.Sampler.from_yaml` method:

.. code-block:: python

    import hydra

    @hydra.main(config_path="conf", config_name="config")
    def my_app(cfg):
      sampler = Sampler.from_yaml(cfg.domain, cfg.sampler)


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

.. autosummary::
  :toctree: _autosummary

    f3dasm.sampling.RandomUniform
    f3dasm.sampling.LatinHypercube
    f3dasm.sampling.SobolSequence


Create your own sampler
--------------------------

Implementing a new sampler works as follows:

* We create a new class inhereting from the :class:`~f3dasm.sampling.sampler.Sampler` class
* We have to implement our own :func:`~f3dasm.sampling.sampler.Sampler.sample_continuous` function:

.. note::

   We can also implement sampling strategies for all the other parameters but this is not necessary

This :func:`~f3dasm.sampling.sampler.Sampler.sample_continuous` function inputs the number of samples you want to create and returns a 2D numpy-array with the coordinates of those samples

.. code-block:: python

   class NewSampler(f3dasm.Sampler):
      def sample_continuous(self, numsamples: int) -> np.ndarray:
         ...

