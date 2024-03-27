.. _data-generation:

Datagenerator
=============

The :class:`~f3dasm.datageneration.DataGenerator` class is the main class of the :mod:`~f3dasm.datageneration` module.
It is used to generate :attr:`~f3dasm.ExperimentData.output_data` for the :class:`~f3dasm.ExperimentData` by taking a :class:`~f3dasm.ExperimentSample` object.

The :class:`~f3dasm.datageneration.DataGenerator` can serve as the interface between the 
:class:`~f3dasm.ExperimentData` object and any third-party simulation software.

.. image:: ../../../img/f3dasm-datageneration.png
    :width: 70%
    :align: center
    :alt: DataGenerator

|

Use the simulator in the data-driven process
--------------------------------------------

In order to run your simulator on each of the :class:`~f3dasm.ExperimentSample` of your :class:`~f3dasm.ExperimentData`, you follow these steps:
In this case, we are utilizing a one of the :ref:`benchmark-functions` to mock a simulator.

We provide the datagenerator to the :meth:`~f3dasm.ExperimentData.evaluate` function with the :class:`~f3dasm.datageneration.DataGenerator` object as an argument.

    .. code-block:: python

        experimentdata.evaluate(data_generator="Ackley", method='sequential', kwargs={'some_additional_parameter': 1})

.. note::

    Any key-word arguments that need to be passed down to the :class:`~f3dasm.datageneration.DataGenerator` can be passed in the :code:`kwargs` argument of the :meth:`~f3dasm.ExperimentData.evaluate` function.


There are three methods available of handeling the :class:`~f3dasm.ExperimentSample` objects:

* :code:`sequential`: regular for-loop over each of the :class:`~f3dasm.ExperimentSample` objects in order
* :code:`parallel`: utilizing the multiprocessing capabilities (with the `pathos <https://pathos.readthedocs.io/en/latest/pathos.html>`_ multiprocessing library), each :class:`~f3dasm.ExperimentSample` object is run in a separate core
* :code:`cluster`: utilizing the multiprocessing capabilities, each :class:`~f3dasm.ExperimentSample` object is run in a separate node. After completion of an sample, the node will automatically pick the next available sample. More information on this mode can be found in the :ref:`cluster-mode` section.
* :code:`cluster_parallel`: Combination of the :code:`cluster` and :code:`parallel` mode. Each node will run multiple samples in parallel.

Implement your simulator
^^^^^^^^^^^^^^^^^^^^^^^^

.. _data-generation-function:

In order to implement your simulator in :mod:`f3dasm`, you need to follow these steps:

1. Create a new class, inheriting from the :class:`~f3dasm.datageneration.DataGenerator` class.
2. Implement the :meth:`~f3dasm.datageneration.DataGenerator.execute` method. This method should not have any arguments (apart from ``self``) and should submit a script with the name of the current ``job_number`` to the simulator.

.. note::

    The :class:`~f3dasm.datageneration.DataGenerator` class has access to the current design through the ``self.experiment_sample`` attribute.
    You can retrieve the ``job_number`` of the current design by calling ``self.experiment_sample.job_number``.



Setting up the pre-processing and post-processing benchmark-functions
---------------------------------------------------------------------

Once you have created the  ``data_generator`` object, you can plug-in a pre-processing and post-processing method:


pre-processing
^^^^^^^^^^^^^^
The preprocessing method is used to create a simulator input file from the information in the :class:`~f3dasm.ExperimentSample`.


This method should adhere to a few things:

* The first argument of the function needs to be ``experiment_sample`` of type :class:`~f3dasm.ExperimentSample`.
* The method should return None.
* The method should create the input file ready for the simulator to process with the job_number as name (``experiment_sample.job_number``) 

You can retrieve the parameters of the :class:`~f3dasm.ExperimentSample` object by calling the :meth:`~f3dasm.ExperimentSample.get` method.

You can add the ``pre-process-function`` to the :class:`~f3dasm.datageneration.DataGenerator` object by passing it through the :meth:`~f3dasm.datageneration.DataGenerator.add_pre_process` method:

.. code-block:: python

    experimentdata.add_pre_process(pre_process_function)

.. note::

    You can add any additional key-word arguments to the :meth:`~f3dasm.datageneration.DataGenerator.add_pre_process` method, which will be passed down to the :meth:`~f3dasm.datageneration.DataGenerator.pre_process` method.


post-processing
^^^^^^^^^^^^^^^

The post-processing method converts the output of the simulator to a ``results.pkl`` `pickle <https://docs.python.org/3/library/pickle.html>`_ file.
This ``results.pkl`` is then loaded into the :class:`~f3dasm.ExperimentData` object.

This method should adhere to a few things:

* The first argument of the function needs to be ``experiment_sample`` of type :class:`~f3dasm.ExperimentSample`.
* The method should return None.
* The method read the output of the simulator (it's name is ``experiment_sample.job_number``) and convert it to a ``results.pkl`` file.
* This pickle file is stored in the current working directory.

You can add the ``post-process-function`` to the :class:`~f3dasm.datageneration.DataGenerator` object by passing it through the :meth:`~f3dasm.datageneration.DataGenerator.add_post_process` method:

.. code-block:: python

    experimentdata.add_post_process(pre_process_function)
