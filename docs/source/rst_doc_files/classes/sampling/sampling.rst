.. _sampling:

Sampling
========

In the context of the data-driven process, samplers play a crucial role in generating input data for experiments or analyses. 
A sampler takes a :class:`~f3dasm.design.Domain` object, and applies a specific strategy to produce samples. 
These samples serve as input for further analysis, experimentation, or modeling.

This section describes how you can implement your sampling strategy or use the built-in samplers in a data-driven process.

.. _integrating-samplers:

Implement your sampling strategy
--------------------------------

When integrating your sampling strategy into the data-driven process, you have to create a function that  will take several arguments:

* :code:`domain`: A :class:`~f3dasm.design.Domain` object that represents the design-of-experiments
* :code:`n_samples`: The number of samples you wish to generate. It's not always necessary to define this upfront, as some sampling methods might inherently determine the number of samples based on certain criteria or properties of the domain.
* :code:`seed`: A seed for the random number generator to replicate the sampling process. This enables you to control the randomness of the sampling process [1]_. By setting a seed, you ensure reproducibility, meaning that if you run the sampling function with the same seed multiple times, you'll get the same set of samples.

.. [1] If no seed is provided, the function should use a random seed.

The function should return the samples (``input_data``) in one of the following formats:

* A :class:`~pandas.DataFrame` object
* A :class:`~numpy.ndarray` object

.. code-block:: python

   def your_sampling_method(domain: Domain, n_samples: int, seed: Optional[int]):
       # Your sampling logic here
       ...
       return your_samples

An example: implementing a sobol sequence sampler
-------------------------------------------------

For example, the following code defines a sampler based on a `Sobol sequence <https://salib.readthedocs.io/en/latest/api/SALib.sample.html?highlight=sobol%20sequence#SALib.sample.sobol_sequence.sample>`_:

.. code-block:: python
   
   from f3dasm.design import Domain
   from SALib.sample import sobol_sequence

   def sample_sobol_sequence(domain: Domain, n_samples: int, **kwargs):
      samples = sobol_sequence.sample(n_samples, len(domain))

      # stretch samples
      for dim, param in enumerate(domain.space.values()):
         samples[:, dim] = (
               samples[:, dim] * (
                  param.upper_bound - param.lower_bound
               ) + param.lower_bound
         )
      return samples

To use the sampler in the data-driven process, you should pass the function to the :class:`~f3dasm.ExperimentData` object as follows:

.. code-block:: python

   from f3dasm.design import ExperimentData, Domain

   domain = Domain(...)
   # Create the ExperimentData object
   experiment_data = ExperimentData(domain=domain)

   # Generate samples
   experiment_data.sample(sampler=your_sampling_method, n_samples=10, seed=42)

.. _implemented samplers:

Implemented samplers
--------------------

The following built-in implementations of samplers can be used in the data-driven process.

======================== ====================================================================== ===========================================================================================================
Name                     Method                                                                 Reference
======================== ====================================================================== ===========================================================================================================
``"random"``             Random Uniform sampling                                                `numpy.random.uniform <https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html>`_
``"latin"``              Latin Hypercube sampling                                               `SALib.latin <https://salib.readthedocs.io/en/latest/api/SALib.sample.html?highlight=latin%20hypercube#SALib.sample.latin.sample>`_
``"sobol"``              Sobol Sequence sampling                                                `SALib.sobol_sequence <https://salib.readthedocs.io/en/latest/api/SALib.sample.html?highlight=sobol%20sequence#SALib.sample.sobol_sequence.sample>`_
``"grid"``               Grid Search sampling                                                   `itertools.product <https://docs.python.org/3/library/itertools.html#itertools.product>`_
======================== ====================================================================== ===========================================================================================================
