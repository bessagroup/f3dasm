Experiment Data
===============

The :class:`~f3dasm.ExperimentData` object is the main object used to store implementations of a design-of-experiments, 
keep track of results, perform optimization and extract data for machine learning purposes.

All other processses of :mod:`f3dasm` use this object to manipulate and access data about your experiments.

The :class:`~f3dasm.ExperimentData` object consists of the following attributes:

- :ref:`domain <domain-format>`: The feasible :class:`~f3dasm.design.Domain` of the Experiment. Used for sampling and optimization.
- :ref:`input_data <input-data-format>`: Tabular data containing the input variables of the experiments as column and the experiments as rows.
- :ref:`output_data <output-data-format>`: Tabular data containing the tracked outputs of the experiments.
- :ref:`project_dir <filename-format>`: A user-defined project directory where all files related to your data-driven process will be stored. 


.. image:: ../../../img/f3dasm-experimentdata.png
    :width: 100%
    :align: center
    :alt: ExperimentData object

|

.. note:: 

    Users of :mod:`f3dasm` are advised to not directly manipulate the attributes of the ExperimentData object. Instead, the methods of ExperimentData should be used to manipulate the data.

The :class:`~f3dasm.ExperimentData` object can be constructed in several ways:

* :ref:`By providing your own data <experimentdata-own>`
* :ref:`Reconstructed from the project directory <experimentdata-file>`
* :ref:`By a sampling strategy <experimentdata-sampling>`
* :ref:`From a hydra configuration file <experimentdata-hydra>`

.. _experimentdata-own:

ExperimentData from your own data
---------------------------------

You can construct a :class:`~f3dasm.ExperimentData` object by providing it :ref:`input_data <input-data-format>`, :ref:`output_data <output-data-format>`, a :ref:`domain <domain-format>` object and a :ref:`filename <filename-format>`.

.. code-block:: python

    >>> from f3dasm import ExperimentData
    >>> data = ExperimentData(
        domain=domain, input_data=input_data, output_data=output_data)


The following sections will explain how to construct a :class:`~f3dasm.ExperimentData` object from your own data.

.. _domain-format:

domain
^^^^^^

The ``domain`` argument should be a :class:`~f3dasm.design.Domain` object. It defines the feasible domain of the design-of-experiments.
Learn more about the :class:`~f3dasm.design.Domain` object in the :ref:`domain <domain>` section.

.. code-block:: python

    >>> from f3dasm import ExperimentData
    >>> from f3dasm.design import Domain
    >>> domain = Domain()
    >>> domain.add_float('x0', 0., 1.)
    >>> domain.add_float('x1', 0., 1.)
    >>> data = ExperimentData(domain=domain)

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

    >>> from f3dasm import ExperimentData
    >>> from f3dasm.design import Domain
    >>> domain.add_float('x0', 0., 1.)
    >>> domain.add_float('x1', 0., 1.)
    >>> data = ExperimentData(domain=domain, input_data=df)

* A two-dimensional :class:`~numpy.ndarray` object with shape (<number of experiments>, <number of input dimensions>)

.. code-block:: python

    >>> from f3dasm import ExperimentData
    >>> from f3dasm.design import Domain
    >>> import numpy as np
    >>> input_array = np.array([[0.1, 0.2], [0.3, 0.4]])
    >>> domain.add_float('x0', 0., 1.)
    >>> domain.add_float('x1', 0., 1.)
    >>> data = ExperimentData(domain=domain, input_data=input_array)

.. note::

    When providing a :class:`~numpy.ndarray` object, you need to provide a :class:`~f3dasm.design.Domain` object as well.
    Also, the order of the input variables is inferred from the order of the columns in the :class:`~f3dasm.design.Domain` object.


* A string or path to a ``.csv`` file containing the input data. The ``.csv`` file should contain a header row with the names of the input variables and the first column should be indices for the experiments.

.. code-block:: python

    >>> from f3dasm import ExperimentData
    >>> from f3dasm.design import Domain
    >>> domain.add_float('x0', 0., 1.)
    >>> domain.add_float('x1', 0., 1.)  
    >>> data = ExperimentData(domain=doman, input_data="my_experiment_data.csv")

.. _output-data-format:

output_data
^^^^^^^^^^^

Output data describes the output variables of the experiments.
The output data is provided in a tabular manner, with the number of rows equal to the number of experiments and the number of columns equal to the number of output variables.


Several datatypes are supported for the ``output_data`` argument:

* A :class:`~pandas.DataFrame` object with the output variable names as columns and the experiments as rows.

    >>> from f3dasm import ExperimentData
    >>> from f3dasm.design import Domain
    >>> df = pd.DataFrame(...) # your data in a pandas DataFrame
    >>> domain.add_output('x0')
    >>> domain.add_output('x1')
    >>> data = ExperimentData(domain=domain, output_data=df)

* A two-dimensional :class:`~numpy.ndarray` object with shape (<number of experiments>, <number of output dimensions>)

    >>> from f3dasm import ExperimentData
    >>> from f3dasm.design import Domain
    >>> import numpy as np
    >>> output_array = np.array([[0.1, 0.2], [0.3, 0.4]])
    >>> domain.add_output('x0')
    >>> domain.add_output('x1')   
    >>> data = ExperimentData(domain=domain, output_array=output_array)

* A string or path to a ``.csv`` file containing the output data. The ``.csv`` file should contain a header row with the names of the output variables and the first column should be indices for the experiments.

    >>> from f3dasm import ExperimentData
    >>> from f3dasm.design import Domain
    >>> domain.add_output('x0')
    >>> domain.add_output('x1') 
    >>> data = ExperimentData(domain=domain, output_data="my_experiment_data.csv")

If you don't have output data yet, you can also construct an :class:`~f3dasm.ExperimentData` object without providing output data.


.. _filename-format:

project directory
^^^^^^^^^^^^^^^^^

The ``project_dir`` argument is used to :ref:`store the ExperimentData to disk <experimentdata-store>`
You can provide a string or a path to a directory. This can either be a relative or absolute path.
If the directory does not exist, it will be created.

.. code-block:: python

    >>> from f3dasm import ExperimentData
    >>> from f3dasm.design import Domain
    >>> project_dir = "folder/to/my_project_directory"
    >>> data = ExperimentData(project_dir=project_dir)

You can also set the project directory manually after creation with the :meth:`~f3dasm.ExperimentData.set_project_dir` method"

.. code-block:: python

    >>> from f3dasm import ExperimentData
    >>> data = ExperimentData()
    >>> data.set_project_dir("folder/to/my_project_directory")


.. _experimentdata-file:

ExperimentData from project directory
-------------------------------------

If you already have constructed the :class:`~f3dasm.ExperimentData` object before, you can retrieve it from disk by calling the :meth:`~f3dasm.ExperimentData.from_file`
classmethod with the path of project directory:

.. code-block:: python

    >>> from f3dasm import ExperimentData
    >>> data = ExperimentData.from_file("folder/to/my_project_directory")

.. _experimentdata-sampling:

ExperimentData from sampling
----------------------------

You can directly construct an :class:`~f3dasm.ExperimentData` object from a sampling strategy by using the :meth:`~f3dasm.ExperimentData.from_sampling` method.
You have to provide the following arguments:

* A sampling function. To learn more about integrating your sampling function, please refer to the :ref:`this <integrating-samplers>` section.
* A :class:`~f3dasm.design.Domain` object describing the input variables of the sampling function.
* The number of samples to generate.
* An optional seed for the random number generator.

.. code-block:: python

    from f3dasm import ExperimentData, Domain

    def your_sampling_function(domain, n_samples, seed):
        # your sampling function
        # ...
        return samples

    domain = Domain()
    domain.add_float('x0', 0., 1.)
    domain.add_float('x1', 0., 1.)  
    data = ExperimentData.from_sampling(sampler=your_sampling_function, domain=domain, n_samples=10, seed=42)

You can use :ref:`built-in samplers <implemented samplers>` by providing one of the following strings as the ``sampler`` argument:

.. code-block:: python

    from f3dasm import ExperimentData
    from f3dasm.design import Domain

    domain = Domain()
    domain.add_float(name='x0', low=0., high=0.)
    domain.add_float(name='x1', low=0., high=0.)
    data = ExperimentData.from_sampling(sampler="latin", domain=domain, n_samples=10, seed=42)

.. _experimentdata-hydra:

ExperimentData from a `hydra <https://hydra.cc/>`_ configuration file
---------------------------------------------------------------------

If you are using `hydra <https://hydra.cc/>`_ for configuring your experiments, you can use it to construct 
an :class:`~f3dasm.ExperimentData` object from the information in the :code:`config.yaml` file with the :meth:`~f3dasm.ExperimentData.from_yaml` method.

You can create an experimentdata :class:`~f3dasm.ExperimentData` object in the same ways as described above, but now using the hydra configuration file.


.. code-block:: yaml
    :caption: config.yaml


    domain:
        x0: 
            type: float
            lower_bound: 0.
            upper_bound: 1.
        x1:
            type: float
            lower_bound: 0.
            upper_bound: 1.

    experimentdata:
        input_data: path/to/input_data.csv
        output_data:
        domain: ${domain}

.. note:: 

    The :class:`~f3dasm.design.Domain` object will be constructed using the :code:`domain` key in the :code:`config.yaml` file. Make sure you have the :code:`domain` key in your :code:`config.yaml`!
    To see how to configure the :class:`~f3dasm.design.Domain` object with hydra, see  :ref:`this <domain-from-yaml>` section.
    
Inside your python script, you can then create the :class:`~f3dasm.ExperimentData` object with the :meth:`~f3dasm.ExperimentData.from_yaml` method:

.. code-block:: python

    >>> from f3dasm import ExperimentData
    >>> import hydra

    >>> @hydra.main(config_path="conf", config_name="config")
    >>> def my_app(config):
    >>>     data = ExperimentData.from_yaml(config.experimentdata)


To create the :class:`~f3dasm.ExperimentData` object with the :meth:`~f3dasm.ExperimentData.from_sampling` method, you can use the following configuration:

.. code-block:: yaml
   :caption: config.yaml for from_sampling

    domain:
        x0: 
            type: float
            lower_bound: 0.
            upper_bound: 1.
        x1:
            type: float
            lower_bound: 0.
            upper_bound: 1.  

    experimentdata:
        from_sampling:
            domain: ${domain}
            sampler: random
            seed: 1
            n_samples: 3


.. note:: 

    Make sure you have the :code:`domain` key in your :code:`config.yaml`!
    To see how to configure the :class:`~f3dasm.design.Domain` object with hydra, see  :ref:`this <domain-from-yaml>` section.


To create the :class:`~f3dasm.ExperimentData` object with the :meth:`~f3dasm.ExperimentData.from_file` method, you can use the following configuration:

.. code-block:: yaml
   :caption: config.yaml for from_file

    experimentdata:
        from_file: path/to/my_experiment_data

Adding data after construction
------------------------------

If you have constructed your :class:`~f3dasm.ExperimentData` object, you can add ``input_data``, ``output_data``, a ``domain`` or the ``filename`` using the :meth:`~f3dasm.ExperimentData.add` method:

.. code-block:: python

    >>> from f3dasm import ExperimentData
    >>> from f3dasm.design import Domain
    >>> data = ExperimentData()
    >>> domain = Domain()
    >>> domain.add_float(name='x0', low=0., high=1.)
    >>> domain.add_float(name='x1', low=0., high=1.)
    >>> data.add(input_data=input_data)

.. warning::

    You can only add data to an existing :class:`~f3dasm.ExperimentData` object if the domain is the same as the existing domain. 


Exporting
---------

.. _experimentdata-store:

Storing the ExperimentData object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~f3dasm.ExperimentData` object can be exported to a collection of files using the :meth:`~f3dasm.ExperimentData.store` method.
You can provide a path to a directory where the files will be stored, or if not provided, the files will be stored in the directory provided in the :attr:`~f3dasm.design.ExperimentData.project_dir` attribute:

.. code-block:: python

    >>> data.store("path/to/project_dir")

Inside the project directory, a subfolder `experiment_data` will be created with the following files:

- :code:`domain.pkl`: The :class:`~f3dasm.design.Domain` object
- :code:`input.csv`: The :attr:`~f3dasm.design.ExperimentData.input_data` table
- :code:`output.csv`: The :attr:`~f3dasm.design.ExperimentData.output_data` table
- :code:`jobs.pkl`: The :attr:`~f3dasm.design.ExperimentData.jobs` object

These files are used to load the :class:`~f3dasm.ExperimentData` object again using the :meth:`~f3dasm.ExperimentData.from_file` method.

.. code-block:: python

    >>> data = ExperimentData.from_file("path/to/project_dir")



.. code-block:: none
   :caption: Directory Structure

   project_dir/
    └── experiment_data/
            ├── domain.pkl
            ├── input.csv
            ├── output.csv
            └── jobs.pkl

.. _experimentdata-store-other:

Storing to other datatypes
^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, you can convert the input- and outputdata of your data-driven process to other well-known datatypes:

* :class:`~numpy.ndarray` (:meth:`~f3dasm.ExperimentData.to_numpy`); creates a tuple of two :class:`~numpy.ndarray` objects containing the input- and outputdata.
* :class:`~xarray.Dataset` (:meth:`~f3dasm.ExperimentData.to_xarray`); creates a :class:`~xarray.Dataset` object containing the input- and outputdata.
* :class:`~pd.DataFrame` (:meth:`~f3dasm.ExperimentData.to_pandas`); creates a tuple of two :class:`~pd.DataFrame` object containing the input- and outputdata.

.. .. minigallery:: f3dasm.ExperimentData
..     :add-heading: Examples using the `ExperimentData` object
..     :heading-level: -

