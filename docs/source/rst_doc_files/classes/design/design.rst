Design
======

A :class:`~f3dasm.design.design.Design` object contains a single realization of the design-of-experiment in :class:`~f3dasm.design.experimentdata.ExperimentData`.

.. image:: ../../../img/f3dasm-design.png
    :alt: Design
    :width: 100%
    :align: center

|

.. note:: 
    A :class:`~f3dasm.design.design.Design` is not constructed manually, but created inside the ExperimentData when it is required by internal processes. 
    The main use of the :class:`~f3dasm.design.design.Design` is to pass it to your own functions and scripts to extract design variables and store output variables.



For each of the experiments in the :class:`~f3dasm.design.experimentdata.ExperimentData`, a :class:`~f3dasm.design.design.Design` object can be created.
This object contains the input and output parameters of a single realization of the :class:`~f3dasm.design.experimentdata.ExperimentData`, as well as the index number of the experiment (:attr:`~f3dasm.design.design.Design.job_number`).

.. code-block:: python
    
   from f3dasm import Design

    def my_function(design: Design, **kwargs):
        parameter1 = design['param_1']
        parameter2 = design['param_2']
        job_number = design.job_number
        ...  # Your own program

        design['output_1'] = output
        return design

A function with a signature like :code:`my_function` can be used as a callable in the :meth:`~f3dasm.design.experimentdata.ExperimentData.run` method to iterate over every design in the :class:`~f3dasm.design.experimentdata.ExperimentData`.

.. note:: 
    In order to use :code:`my_function` within :mod:`f3dasm` workflow, the first argument needs to be a :class:`~f3dasm.design.design.Design` object. 
    The function can have any number of additional arguments, which will be passed to the function when it is called.
    Lastly, the :class:`~f3dasm.design.design.Design` must be returned.

Extract parameters from a design
--------------------------------

Input parameters of a design can be accessed using the :code:`[]` operator, with the name of the parameter as the key.
Only input parameters of the design can be accessed this way, and an error will be raised if the key is not found.

.. code-block:: python

    >>> design['param_1']
    0.0249


The job_number of the design can be accessed using the :attr:`~f3dasm.design.design.Design.job_number` attribute and is zero-indexed.

.. code-block:: python

    >>> design.job_number
    0

The input and output parameters of a design can be extracted as a tuple of numpy arrays with the :meth:`~f3dasm.design.Design.to_numpy` method.

.. code-block:: python

    >>> design.to_numpy()
    (np.array([0.0249, 0.034, 0.100]), np.array([]))

Storing output parameters to the design
---------------------------------------

After running your calculation, you can store the result back into the design in two ways:

* Singular values and small lists can be stored directly to the :attr:`~f3dasm.design.experimentdata.ExperimentData.output_data`
* Large objects can be stored to disk with the :meth:`f3dasm.design.design.Design.store` method.

Single values or small lists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Single values or small lists can be stored to the design using the :code:`[]` operator, with the name of the parameter as the key. 
This will create a new output parameter if the parameter name is not found in :attr:`~f3dasm.design.experimentdata.ExperimentData.output_data` of the :class:`~f3dasm.design.experimentdata.ExperimentData`.

.. code-block:: python

    >>> design['output_1'] = 0.123
    >>> design['output_2'] = [0.123, 0.456, 0.789]
    >>> design['output_3'] = 'Hello world'

All built-in types are supported for storing to the design this way. Array-like data such as numpy arrays and pandas dataframes are **not** supported and will raise an error.

.. note:: 
    Outputs stored directly to the :attr:`~f3dasm.design.experimentdata.ExperimentData.output_data` will be stored within the :class:`~f3dasm.design.experimentdata.ExperimentData` object.
    This means that the output will be loaded into memory everytime this object is accessed. For large outputs, it is recommended to store the output to disk. 

Large objects and array-like data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to store large objects or array-like data, the :meth:`~f3dasm.design.design.Design.store` method can be used. A reference (:code:`Path`) will be saved to the :attr:`~f3dasm.design.experimentdata.ExperimentData.output_data`.

.. code-block:: python

    >>> design.store('output_1', my_large_object)

:mod:`f3dasm` will automatically create a new directory for each output parameter and store the object with a generated filename referencing the :attr:`~f3dasm.design.design.Design.job_number` of the design.

.. code-block:: none
   :caption: Directory Structure

   my_project/
   ├── output_1/
   │   ├── 0.npy
   │   ├── 1.npy
   │   ├── 2.npy
   │   └── 3.npy
   ├── my_experiment_domain.pkl
   ├── my_experiment_data.csv
   ├── my_experiment_output.csv
   └── my_experiment_jobs.pkl

In the :attr:`~f3dasm.design.experimentdata.ExperimentData.output_data`, a reference to the stored object (e.g. :code:`my_project/output_1/0.npy`) will be automatically appended to the `<output parameter name>_path` parameter.

.. code-block:: python

    >>> design['output_1_path']
    'my_project/output_1/0.npy'



:mod:`f3dasm` has built-in storing functions for numpy arrays, pandas DataFrames and xarray DataArrays and Datasets. 
For any other type of object, you can provide a storing function to the :meth:`~f3dasm.design.design.Design.store` method call:

* The arguments must be the object itself and the path that it should store to
* The function should store the object to disk.
* The return value must be the file extension of the stored object as a string.

You can take the following function as an example:

.. code-block:: python

    def numpy_storing_function(object, path: Path) -> str:
        np.save(file=path.with_suffix('.npy'), arr=object)
        return '.npy'


After defining the storing function, it can be used as a callable in the :meth:`~f3dasm.design.design.Design.store` method:

.. code-block:: python

    >>> design.store('output_1', my_large_object, numpy_storing_function)