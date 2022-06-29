The design space
----------------

Constructing the design space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are three types of parameters that can be created: continous, discrete and categorical:

* We can create **continous** parameters with a :attr:`~f3dasm.src.space.ContinuousSpace.lower_bound` and :attr:`f3dasm.src.space.ContinuousSpace.upper_bound` with the :class:`~f3dasm.src.space.ContinuousSpace` class
* We can create **discrete** parameters with a :attr:`~f3dasm.src.space.ContinuousSpace.lower_bound` and :attr:`~f3dasm.src.space.ContinuousSpace.upper_bound` with the :class:`~f3dasm.src.space.DiscreteSpace` class
* We can create **categorical** parameters with a list of strings (:attr:`~f3dasm.src.space.CategoricalSpace.categories`) with the :attr:`~f3dasm.src.space.CategoricalSpace` class

The design space is then constructed by calling the :class:`~f3dasm.src.designofexperiments.DoE` class and providing either an list of parameters (:attr:`~f3dasm.src.designofexperiments.DoE.input_space`):