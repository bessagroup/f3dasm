Generating output using the Data Generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~f3dasm.datageneration.DataGenerator` class is the main class of the :mod:`~f3dasm.datageneration` module.
It is used to generate :attr:`~f3dasm.ExperimentData.output_data` for the :class:`~f3dasm.ExperimentData`.

The :class:`~f3dasm.datageneration.DataGenerator` can serve as the interface between the 
:class:`~f3dasm.ExperimentData` object and any third-party simulation software.