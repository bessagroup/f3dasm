Sampling
========

Samplers take the :class:`~f3dasm.design.Domain` object and return input data  based on the sampling strategy.

.. _integrating-samplers:

Implement your sampling strategy
--------------------------------

To integrate your sampling strategy in in the data-driven process, you should create a function that takes the following arguments:

* A :class:`~f3dasm.design.Domain` object
* The number of samples to create
* A random seed (optional)

The function should return the samples (``input_data``) in one of the following formats:

* A :class:`~pandas.DataFrame` object
* A :class:`~numpy.ndarray` object

.. note::
   
   ...


.. _implemented samplers:

Use the sampler in the data-driven process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use the sampler in the data-driven process, you should pass the function to the :class:`~f3dasm.design.ExperimentData` object as follows:

.. code-block:: python

   from f3dasm.design import ExperimentData, Domain
   from 

   domain = Domain(...)
   # Create the ExperimentData object
   experiment_data = ExperimentData(domain=domain)

   # Generate samples
   experiment_data.sample(sampler=your_sampling_method, n_samples=10, seed=42)


.. note::

   This method will throw an error if you do not have any prior ``input_data`` in the :class:`~f3dasm.design.ExperimentData` 
   object before sampling **and** you do not provide a :class:`~f3dasm.design.Domain` object in the initializer.

Implemented samplers
--------------------

The following built-in implementations of samplers can be used in the data-driven process.
To use these samplers

======================== ====================================================================== ===========================================================================================================
Name                     Method                                                                 Reference
======================== ====================================================================== ===========================================================================================================
``"random"``             Random Uniform sampling                                                `numpy.random.uniform <https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html>`_
``"latin"``              Latin Hypercube sampling                                               `SALib.latin <https://salib.readthedocs.io/en/latest/api/SALib.sample.html?highlight=latin%20hypercube#SALib.sample.latin.sample>`_
``"sobol"``              Sobol Sequence sampling                                                `SALib.sobol_sequence <https://salib.readthedocs.io/en/latest/api/SALib.sample.html?highlight=sobol%20sequence#SALib.sample.sobol_sequence.sample>`_
======================== ====================================================================== ===========================================================================================================
