Experiment Data
===============

The :class:`~f3dasm.design.experimentdata.ExperimentData` object is the main object used to store implementations of a design-of-experiments, 
keep track of results, perform optimization and extract data for machine learning purposes.

All other processses of f3dasm use this object to manipulate and access data about your experiments.

The :class:`~f3dasm.design.experimentdata.ExperimentData` object consists of the following attributes:

- :attr:`~f3dasm.design.experimentdata.ExperimentData.filename`: Name of the ExperimentData, used for storing and loading.
- :attr:`~f3dasm.design.experimentdata.ExperimentData.domain`: The feasible :class:`~f3dasm.design.domain.Domain` of the Experiment. Used for sampling and optimization.
- :attr:`~f3dasm.design.experimentdata.ExperimentData.input_data`: Tabular data containing the input variables of the experiments as column and the experiments as rows.
- :attr:`~f3dasm.design.experimentdata.ExperimentData.output_data`: Tabular data containing the tracked outputs of the experiments.
- :attr:`~f3dasm.design.experimentdata.ExperimentData.jobs`: Index-like object tracking if experiments have been executed.

.. image:: ../../img/f3dasm-experimentdata.png
    :width: 100%
    :align: center
    :alt: ExperimentData object

|

.. note:: 

    Users of :code:`f3dasm` are advised to not directly manipulate the attributes of the ExperimentData object. Instead, the methods of ExperimentData should be used to manipulate the data.

Constructing
------------

The default constructor (:meth:`~f3dasm.design.experimentdata.ExperimentData.__init__`) requires a :class:`~f3dasm.design.domain.Domain` object and a name.
It will construct an empty :class:`~f3dasm.design.experimentdata.ExperimentData` object with the given name and domain.

If you already have a source of data, there are alternative ways to construct an :class:`~f3dasm.design.experimentdata.ExperimentData` object:

From a file containing a serialized :class:`~f3dasm.design.experimentdata.ExperimentData` object: :meth:`~f3dasm.design.experimentdata.ExperimentData.from_file`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you already have constructed the :class:`~f3dasm.design.experimentdata.ExperimentData` object before, you can retrieve it from disk by calling the :meth:`~f3dasm.design.experimentdata.ExperimentData.from_file`
method. 

.. code-block:: python

    >>> from f3dasm import ExperimentData
    >>> data = ExperimentData.from_file("my_experiment")

From a :class:`~f3dasm.sampling.sampler.Sampler`: :meth:`~f3dasm.design.experimentdata.ExperimentData.from_sampling`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sample a :class:`~f3dasm.design.experimentdata.ExperimentData` object from a :class:`~f3dasm.sampling.sampler.Sampler` object.
You can use the built-in samplers from the sampling module or construct your own.

.. code-block:: python

    >>> from f3dasm import ExperimentData, Domain, ContinuousParameter
    >>> from f3dasm.sampling import RandomUniform
    >>> domain = Domain({'x0': ContinuousParameter(0., 1.)}, 'x1': ContinuousParameter(0., 1.)}
    >>> sampler = RandomUniform(domain, 10)
    >>> data = ExperimentData.from_sampling(sampler)

From a csv file: :meth:`~f3dasm.design.experimentdata.ExperimentData.from_csv`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you already have already created realizations of your design-of-experiments, you can load them from a csv file.
The csv file should contain a header row with the names of the input variables and the first column should be indices for the experiments.

Additionally, you can provide the :class:`~f3dasm.design.domain.Domain` object that suits your design-of-experiments.


.. code-block:: python

    >>> from f3dasm import ExperimentData, Domain
    >>> domain = Domain({'x0': ContinuousParameter(0., 1.)}, 'x1': ContinuousParameter(0., 1.)})    
    >>> data = ExperimentData.from_csv("my_experiment_data.csv", domain)

.. note:: 

    If you don't provide a suitable Domain object, a Domain will be inferred from the input data.

From a pandas DataFrame: :meth:`~f3dasm.design.experimentdata.ExperimentData.from_dataframe`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you already have already created realizations of your design-of-experiments, you can also load them from a pandas DataFrame.
The pandas DataFrame should contain a header row with the names of the input variables and indices for the experiments.

Additionally, you can provide the :class:`~f3dasm.design.domain.Domain` object that suits your design-of-experiments.

.. code-block:: python

    >>> from f3dasm import ExperimentData, Domain
    >>> df = pd.DataFrame(...) # your data in a pandas DataFrame
    >>> domain = Domain({'x0': ContinuousParameter(0., 1.)}, 'x1': ContinuousParameter(0., 1.)})    
    >>> data = ExperimentData.from_dataframe(df, domain)

From a hydra YAML-configuration file: :meth:`~f3dasm.design.experiementdata.ExperimentData.from_yaml`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are using hydra for configuring your expeirments, you can use it to construct 
an :class:`~f3dasm.design.experimentdata.ExperimentData` object from the information in the config file.

.. code-block:: python

    >>> from f3dasm import ExperimentData
    >>> import hydra

    >>> @hydra.main(config_path="conf", config_name="config")
    >>> def my_app(cfg):
    >>>     data = ExperimentData.from_yaml(cfg)

For more information on how to use hydra with f3dasm, please refer to the experimentdata section.

Exporting
---------

The :class:`~f3dasm.design.experimentdata.ExperimentData` object can be exported to a file using the :meth:`~f3dasm.design.experimentdata.ExperimentData.store` method.
This will create a series of files containing its attributes:

- :code:`<filename>_domain.pkl`: The :class:`~f3dasm.design.domain.Domain` object
- :code:`<filename>_data.csv`: The :attr:`~f3dasm.design.experimentdata.ExperimentData.input_data` table
- :code:`<filename>_output.csv`: The :attr:`~f3dasm.design.experimentdata.ExperimentData.output_data` table
- :code:`<filename>_jobs.pkl`: The :attr:`~f3dasm.design.experimentdata.ExperimentData.jobs` object

These files can be used to load the :class:`~f3dasm.design.experimentdata.ExperimentData` object again using the :meth:`~f3dasm.design.experimentdata.ExperimentData.from_file` method.

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


Alternatively, you can convert the input- and outputdata to numpy arrays (:meth:`~f3dasm.design.experimentdata.ExperimentData.to_numpy`) or xarray (:meth:`~f3dasm.design.experimentdata.ExperimentData.to_xarray`).