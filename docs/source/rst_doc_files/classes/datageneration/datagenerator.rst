.. _data-generation:

Datagenerator
=============

The :class:`~f3dasm.datageneration.DataGenerator` class is the main class of the :mod:`~f3dasm.datageneration` module.
It is used to generate :attr:`~f3dasm.design.ExperimentData.output_data` for the :class:`~f3dasm.design.ExperimentData` by taking a :class:`~f3dasm.design.ExperimentSample` object.

The :class:`~f3dasm.datageneration.DataGenerator` can serve as the interface between the 
:class:`~f3dasm.design.ExperimentData` object and any third-party simulation software.

.. image:: ../../../img/f3dasm-datageneration.png
    :width: 70%
    :align: center
    :alt: DataGenerator

|

Creating a data-generator
-------------------------

In order to run your simulator on each of the :class:`~f3dasm.design.ExperimentSample` of your :class:`~f3dasm.design.ExperimentData`, you follow these steps:
In this case, we are utilizing a one of the :ref:`benchmark-functions` to mock a simulator.

1. Construct the :class:`~f3dasm.datageneration.DataGenerator` object.

    .. code-block:: python

        from f3dasm.datageneration.functions import Ackley
        datagenerator = Ackley()

2. Call the :meth:`~f3dasm.design.ExperimentData.run` function with the :class:`~f3dasm.datageneration.DataGenerator` object's :meth:`~f3dasm.datageneration.DataGenerator.run` method as an argument.

    .. code-block:: python

        experimentdata.run(datagenerator.run, method='sequential', kwargs={'some_kwarg': 1})

.. note::

    Any key-word arguments that need to be passed down to the :class:`~f3dasm.datageneration.DataGenerator` :code:`__call__` function can be passed in the :code:`kwargs` argument of the :meth:`~f3dasm.design.ExperimentData.run` function.


There are three methods available of handeling the :class:`~f3dasm.design.ExperimentSample` objects:

* :code:`sequential`: regular for-loop over each of the :class:`~f3dasm.design.ExperimentSample` objects in order
* :code:`parallel`: utilizing the multiprocessing capabilities, each :class:`~f3dasm.design.ExperimentSample` object is run in a separate core
* :code:`cluster`: utilizing the multiprocessing capabilities, each :class:`~f3dasm.design.ExperimentSample` object is run in a separate node. After completion of an sample, the node will automatically pick the next available sample. More information on this mode can be found in the :ref:`cluster-mode` section.


Implemented data-generators
---------------------------

For a list of all the benchmark functions that are implemented, see :ref:`benchmark-functions`.

======================== ========================================================================= ===============================================================================================
Name                      Docs of the Python class                                                 Reference
======================== ========================================================================= ===============================================================================================
Example                  :code:`example`                                                            `scipy.minimize CG <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cg.html>`_
======================== ========================================================================= ===============================================================================================


Extend the simulator capabilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :mod:`f3dasm.datageneration` module is designed to be easily extended by third-party libraries.
In order to not bloat the main :mod:`f3dasm` package, these extensions are provided as separate package: `f3dasm_simulate <https://github.com/bessagroup/f3dasm_simulate>`_.

More information can be found in the :ref:`f3dasm-simulate` section.

.. _data-generation-function:

Create your own data-generator
------------------------------

In order to use your own simulator or script, you need to comply with either one of the following options:

* Create a class that inherits from the :class:`~f3dasm.datageneration.DataGenerator` class and implement the methods.
* Create a function that takes a :class:`~f3dasm.design.ExperimentSample` object as an argument (and returns a :class:`~f3dasm.design.ExperimentSample`).


Inherit from DataGenerator
^^^^^^^^^^^^^^^^^^^^^^^^^^

The datagenerator protocol class has three main methods:

* :meth:`~f3dasm.datageneration.DataGenerator.pre_process` - Any preprocessing done before the simulation, optional. Returns None
* :meth:`~f3dasm.datageneration.DataGenerator.execute` - The main functon call to run the simulation. Returns None.
* :meth:`~f3dasm.datageneration.DataGenerator.post_process` - Any post-processing done after the simulation, optional. Returns None

.. note::

    The :meth:`~f3dasm.datagenerationr.DataGenerator.run` method chains these three methods together and is inherited from the parent class.


In order to create your own data-generator, you need to 

1. inherit from the :class:`~f3dasm.datageneration.DataGenerator` class.
2. implement at least the :meth:`~f3dasm.datageneration.DataGenerator.execute` method, the :meth:`~f3dasm.datageneration.DataGenerator.pre_process` and :meth:`~f3dasm.datageneration.DataGenerator.post_process` are optional.

The the :meth:`~f3dasm.datageneration.DataGenerator.pre_process`, :meth:`~f3dasm.datageneration.DataGenerator.execute` and :meth:`~f3dasm.datageneration.DataGenerator.post_process` have to accept arbitrary key-word arguments (:code:`**kwargs`). 
This is because the :meth:`~f3dasm.datageneration.DataGenerator.run` method will pass any key-word arguments to these methods.

An example is given in the following code block:

.. code-block:: python

    class MySimulator(DataGenerator):
    def __init__(self, simulator_specific_parameters: Any):
        self.simulator_specific_parameters = simulator_specific_parameters

    def pre_process(self, any_pre_process_arg: str, **kwargs) -> None:
        ...

    def execute(self, any_argument: str, **kwargs) -> None:
        # Retrieve parameters
        parameter_1 = self.experiment_sample['parameter1']

        # Run a simulation
        ...

        # Store the results
        self.experiment_sample['result'] = result

    def post_process(self, any_post_process_arg: str, **kwargs) -> None:
        ...

In order to run this simulator, you need to create an instance of the class and pass it to the :meth:`~f3dasm.design.ExperimentData.run` function:

.. code-block:: python

    simulator = MySimulator(simulator_specific_parameters)
    experimentdata.run(simulator.run, method='sequential', kwargs={'any_pre_process_arg': 'pre', 'any_argument': 'arg', 'any_post_process_arg': 'post'})

Create a data-generator from a functional approach
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The functional approach is a bit more flexible, as it allows you to use any function 
that takes a :class:`~f3dasm.design.ExperimentSample` object as an argument, and returns a :class:`~f3dasm.design.ExperimentSample` object.

.. note::

    The :class:`~f3dasm.datageneration.DataGenerator` class is a wrapper around the functional approach.


.. code-block:: python
    
    from f3dasm import ExperimentSample

    def my_function(experiment_sample: f3dasm.ExperimentSample, some_kwarg: int):
        # do something with the sample
        return experiment_sample

    experimentdata.run(my_function, method='sequential', kwargs={'some_kwarg': 1})