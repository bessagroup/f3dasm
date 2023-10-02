Experiment Data
===============

The :class:`~f3dasm.design.ExperimentData` object is the main object used to store implementations of a design-of-experiments, 
keep track of results, perform optimization and extract data for machine learning purposes.

All other processses of f3dasm use this object to manipulate and access data about your experiments.

The :class:`~f3dasm.design.ExperimentData` object consists of the following attributes:

- :ref:`domain <domain-format>`: The feasible :class:`~f3dasm.design.Domain` of the Experiment. Used for sampling and optimization.
- :ref:`input_data <input-data-format>`: Tabular data containing the input variables of the experiments as column and the experiments as rows.
- :ref:`output_data <output-data-format>`: Tabular data containing the tracked outputs of the experiments.
- :ref:`filename <filename-format>`: Name of the ExperimentData, used for storing and loading.


.. image:: ../../../img/f3dasm-experimentdata.png
    :width: 100%
    :align: center
    :alt: ExperimentData object

|

.. note:: 

    Users of :mod:`f3dasm` are advised to not directly manipulate the attributes of the ExperimentData object. Instead, the methods of ExperimentData should be used to manipulate the data.

The :class:`~f3dasm.design.ExperimentData` object can be constructed in several ways:

* :ref:`By providing your own data <experimentdata-own>`
* :ref:`Retrieved from disk <experimentdata-file>`
* :ref:`By a sampling strategy <experimentdata-sampling>`
* :ref:`From a hydra configuration file <experimentdata-hydra>`

.. _experimentdata-own:

ExperimentData from your own data
---------------------------------

You can construct a :class:`~f3dasm.design.ExperimentData` object by providing it :ref:`input_data <input-data-format>`, :ref:`output_data <output-data-format>`, a :ref:`domain <domain-format>` object and a :ref:`filename <filename-format>`.

.. code-block:: python

    >>> from f3dasm import ExperimentData
    >>> data = ExperimentData(domain, input_data, output_data)


The following sections will explain how to construct a :class:`~f3dasm.design.ExperimentData` object from your own data.

.. _domain-format:

domain
^^^^^^

The ``domain`` argument should be a :class:`~f3dasm.design.Domain` object. It defines the feasible domain of the design-of-experiments.
Learn more about the :class:`~f3dasm.design.Domain` object in the :ref:`domain <domain>` section.

.. code-block:: python

    >>> from f3dasm import ExperimentData, Domain
    >>> domain = Domain()
    >>> domain.add_float('x0', 0., 1.)
    >>> domain.add_float('x1', 0., 1.)
    >>> data = ExperimentData(domain)

.. warning ::

    If you don't provide a :class:`~f3dasm.design.Domain` object, the domain will be inferred from the input data. 
    Constructing the dataframe by inferring it from samples can be useful if you have a large number of parameters and you don't want to manually specify the domain.
    This will be done by looking at the data-type and boundaries of the input data. 
    However, this is not recommended as it can lead to unexpected results.

.. _input-data-format:

input_data
^^^^^^^^^^

Input data describes the input variables of the experiments. 
The input data is provided in a tabular manner, with the number of rows equal to the number of experiments and the number of columns equal to the number of input variables.

Single parameter values can have any of the basic built-in types: ``int``, ``float``, ``str``, ``bool``. Lists, tuples or array-like structures are not allowed.

Several datatypes are supported for the ``input_data`` argument:

* A :class:`~pandas.DataFrame` object with the input variable names as columns and the experiments as rows.

.. code-block:: python

    >>> from f3dasm import ExperimentData, Domain
    >>> df = pd.DataFrame(...) # your data in a pandas DataFrame
    >>> domain = Domain({'x0': ContinuousParameter(0., 1.)}, 'x1': ContinuousParameter(0., 1.)})    
    >>> data = ExperimentData.from_dataframe(df, domain)

* A two-dimensional :class:`~numpy.ndarray` object with shape (<number of experiments>, <number of input dimensions>)

.. code-block:: python

    >>> from f3dasm import ExperimentData, Domain
    >>> import numpy as np
    >>> input_data = np.array([[0.1, 0.2], [0.3, 0.4]])
    >>> domain = Domain({'x0': ContinuousParameter(0., 1.)}, 'x1': ContinuousParameter(0., 1.)})    
    >>> data = ExperimentData.from_array(input_data, domain)

.. note::

    When providing a :class:`~numpy.ndarray` object, you need to provide a :class:`~f3dasm.design.Domain` object as well.
    Also, the order of the input variables is inferred from the order of the columns in the :class:`~f3dasm.design.Domain` object.


* A string or path to a ``.csv`` file containing the input data. The ``.csv`` file should contain a header row with the names of the input variables and the first column should be indices for the experiments.

.. code-block:: python

    >>> from f3dasm import ExperimentData, Domain
    >>> domain = Domain({'x0': ContinuousParameter(0., 1.)}, 'x1': ContinuousParameter(0., 1.)})    
    >>> data = ExperimentData.from_csv("my_experiment_data.csv", domain)

.. _output-data-format:

output_data
^^^^^^^^^^^

Output data describes the output variables of the experiments.
The output data is provided in a tabular manner, with the number of rows equal to the number of experiments and the number of columns equal to the number of output variables.


Several datatypes are supported for the ``output_data`` argument:

* A :class:`~pandas.DataFrame` object with the output variable names as columns and the experiments as rows.

    >>> from f3dasm import ExperimentData, Domain
    >>> df = pd.DataFrame(...) # your data in a pandas DataFrame
    >>> domain = Domain({'x0': ContinuousParameter(0., 1.)}, 'x1': ContinuousParameter(0., 1.)})    
    >>> data = ExperimentData.from_dataframe(df, domain)

* A two-dimensional :class:`~numpy.ndarray` object with shape (<number of experiments>, <number of output dimensions>)

    >>> from f3dasm import ExperimentData, Domain
    >>> import numpy as np
    >>> input_data = np.array([[0.1, 0.2], [0.3, 0.4]])
    >>> domain = Domain({'x0': ContinuousParameter(0., 1.)}, 'x1': ContinuousParameter(0., 1.)})    
    >>> data = ExperimentData.from_array(input_data, domain)

* A string or path to a ``.csv`` file containing the output data. The ``.csv`` file should contain a header row with the names of the output variables and the first column should be indices for the experiments.

    >>> from f3dasm import ExperimentData, Domain
    >>> domain = Domain({'x0': ContinuousParameter(0., 1.)}, 'x1': ContinuousParameter(0., 1.)})    
    >>> data = ExperimentData.from_csv("my_experiment_data.csv", domain)

If you don't have output data yet, you can also construct an :class:`~f3dasm.design.ExperimentData` object without providing output data.


.. _filename-format:

filename
^^^^^^^^

The ``filename`` argument is optional and can be used to :ref:`store the ExperimentData to disk <experimentdata-store>`
You can provide a string or a path to a file.

.. code-block:: python

    >>> from f3dasm import ExperimentData, Domain
    >>> filename = "folder/to/my_experiment_data"
    >>> data = ExperimentData(filename=filename)

.. _experimentdata-file:

ExperimentData from a file containing a serialized :class:`~f3dasm.design.ExperimentData` object
------------------------------------------------------------------------------------------------

If you already have constructed the :class:`~f3dasm.design.ExperimentData` object before, you can retrieve it from disk by calling the :meth:`~f3dasm.design.ExperimentData.from_file`
method with the path of the files. 

.. code-block:: python

    >>> from f3dasm import ExperimentData
    >>> data = ExperimentData.from_file("my_experiment")

.. _experimentdata-sampling:

ExperimentData from a sampling
------------------------------

You can directly construct an :class:`~f3dasm.design.ExperimentData` object from a sampling strategy by using the :meth:`~f3dasm.design.ExperimentData.from_sampling` method.
You have to provide the following arguments:

* A sampling function. To learn more about integrating your sampling function, please refer to :ref:`this <integrating-sampling>` section.
* A :class:`~f3dasm.design.Domain` object describing the input variables of the sampling function.
* The number of samples to generate.
* An optional seed for the random number generator.

.. code-block:: python

    from f3dasm import ExperimentData, Domain, ContinuousParameter

    def your_sampling_function(domain, n_samples, seed):
        # your sampling function
        # ...
        return samples

    domain = Domain({'x0': ContinuousParameter(0., 1.)}, 'x1': ContinuousParameter(0., 1.)}
    sampler = RandomUniform(domain, 10)
    data = ExperimentData.from_sampling(sampler=your_sampling_function, domain=domain, n_samples=10, seed=42)

You can use the built-in samplers from the sampling module by providing one of the following strings as the ``sampler`` argument:

======================== ====================================================================== ===========================================================================================================
Name                     Method                                                                 Reference
======================== ====================================================================== ===========================================================================================================
``"random"``             Random Uniform sampling                                                `numpy.random.uniform <https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html>`_
``"latin"``              Latin Hypercube sampling                                               `SALib.latin <https://salib.readthedocs.io/en/latest/api/SALib.sample.html?highlight=latin%20hypercube#SALib.sample.latin.sample>`_
``"sobol"``              Sobol Sequence sampling                                                `SALib.sobol_sequence <https://salib.readthedocs.io/en/latest/api/SALib.sample.html?highlight=sobol%20sequence#SALib.sample.sobol_sequence.sample>`_
======================== ====================================================================== ===========================================================================================================

.. code-block:: python

    from f3dasm import ExperimentData, Domain, ContinuousParameter

    domain = Domain({'x0': ContinuousParameter(0., 1.)}, 'x1': ContinuousParameter(0., 1.)}
    data = ExperimentData.from_sampling(sampler="latin", domain=domain, n_samples=10, seed=42)

.. _experimentdata-hydra:

ExperimentData from a `hydra <https://hydra.cc/>`_ configuration file
---------------------------------------------------------------------

If you are using `hydra <https://hydra.cc/>`_ for configuring your experiments, you can use it to construct 
an :class:`~f3dasm.design.ExperimentData` object from the information in the :code:`config.yaml` file with the :meth:`~f3dasm.design.ExperimentData.from_yaml` method.

You can create an experimentdata :class:`~f3dasm.design.ExperimentData` object in the same ways as described above, but now using the hydra configuration file.


.. code-block:: yaml
    :caption: config.yaml


    domain:
        x0: 
            _target_: f3dasm.ContinuousParameter
            lower_bound: 0.
            upper_bound: 1.
        x1:
            _target_: f3dasm.ContinuousParameter
            lower_bound: 0.
            upper_bound: 1.

    experimentdata:
        input_data: path/to/input_data.csv
        output_data:
        domain:  ${domain}

.. note:: 

    The :class:`~f3dasm.design.Domain` object will be constructed using the :code:`domain` key in the :code:`config.yaml` file. Make sure you have the :code:`domain` key in your :code:`config.yaml`!
    To see how to configure the :class:`~f3dasm.design.Domain` object with hydra, see  :ref:`this <domain-from-yaml>` section.
    
Inside your python script, you can then create the :class:`~f3dasm.design.ExperimentData` object with the :meth:`~f3dasm.design.ExperimentData.from_yaml` method:

.. code-block:: python

    >>> from f3dasm import ExperimentData
    >>> import hydra

    >>> @hydra.main(config_path="conf", config_name="config")
    >>> def my_app(config):
    >>>     data = ExperimentData.from_yaml(config)

.. note:: 

    Make sure to pass the full :code:`config` to the :meth:`~f3dasm.design.ExperimentData.from_yaml` constructor!

To create the :class:`~f3dasm.design.ExperimentData` object with the :meth:`~f3dasm.design.ExperimentData.from_sampling` method, you can use the following configuration:

.. code-block:: yaml
   :caption: config.yaml for from_sampling

    domain:
        x0: 
            _target_: f3dasm.ContinuousParameter
            lower_bound: 0.
            upper_bound: 1.
        x1:
            _target_: f3dasm.ContinuousParameter
            lower_bound: 0.
            upper_bound: 1.    

    experimentdata:
        from_sampling:
            _target_: f3dasm.sampling.RandomUniform
            seed: 1
            number_of_samples: 3


.. note:: 

    The :class:`~f3dasm.sampling.Sampler` object will be constructed using the :class:`~f3dasm.design.Domain` object from the config file. Make sure you have the :code:`domain` key in your :code:`config.yaml`!
    To see how to configure the :class:`~f3dasm.design.Domain` object with hydra, see  :ref:`this <domain-from-yaml>` section.


To create the :class:`~f3dasm.design.ExperimentData` object with the :meth:`~f3dasm.design.ExperimentData.from_file` method, you can use the following configuration:

.. code-block:: yaml
   :caption: config.yaml for from_file

    experimentdata:
        from_file: path/to/my_experiment_data

Adding data after construction
------------------------------

If you have constructed your :class:`~f3dasm.design.ExperimentData` object, you can add ``input_data``, ``output_data``, a ``domain`` or the ``filename`` using the :meth:`~f3dasm.design.ExperimentData.add` method:

.. code-block:: python

    >>> from f3dasm import ExperimentData, Domain
    >>> data = ExperimentData()
    >>> domain = Domain({'x0': ContinuousParameter(0., 1.)}, 'x1': ContinuousParameter(0., 1.)}
    >>> data.add(input_data, output_data, domain, filename)

.. warning::

    You can only add data to an existing :class:`~f3dasm.design.ExperimentData` object if the domain is the same as the existing domain. 


Exporting
---------

.. _experimentdata-store:

Storing the ExperimentData object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~f3dasm.design.ExperimentData` object can be exported to a file using the :meth:`~f3dasm.design.ExperimentData.store` method.
This will create a series of files containing its attributes:

- :code:`<filename>_domain.pkl`: The :class:`~f3dasm.design.Domain` object
- :code:`<filename>_data.csv`: The :attr:`~f3dasm.design.ExperimentData.input_data` table
- :code:`<filename>_output.csv`: The :attr:`~f3dasm.design.ExperimentData.output_data` table
- :code:`<filename>_jobs.pkl`: The :attr:`~f3dasm.design.ExperimentData.jobs` object

These files can be used to load the :class:`~f3dasm.design.ExperimentData` object again using the :meth:`~f3dasm.design.ExperimentData.from_file` method.

.. code-block:: python

    >>> from f3dasm import ExperimentData
    >>> data = ExperimentData.from_file("my_experiment")
    >>> data.store("my_experiment")

This will result in the creation of the following files:

.. code-block:: none
   :caption: Directory Structure

   my_project/
   ├── my_experiment_domain.pkl
   ├── my_experiment_data.csv
   ├── my_experiment_output.csv
   └── my_experiment_jobs.pkl

.. _experimentdata-store-other:

Storing to other datatypes
^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, you can convert the input- and outputdata of your data-driven process to other well-knowndatatypes:

* :class:`~numpy.ndarray` (:meth:`~f3dasm.design.ExperimentData.to_numpy`); creates a tuple of two :class:`~numpy.ndarray` objects containing the input- and outputdata.
* :class:`~xarray.Dataset` (:meth:`~f3dasm.design.ExperimentData.to_xarray`); creates a :class:`~xarray.Dataset` object containing the input- and outputdata.
* :class:`~pd.DataFrame` (:meth:`~f3dasm.design.ExperimentData.to_pandas`); creates a tuple of two :class:`~pd.DataFrame` object containing the input- and outputdata.
