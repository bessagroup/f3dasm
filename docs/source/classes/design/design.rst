Design space
============

The design space can be created with the :class:`~f3dasm.design.design.DesignSpace` class.

Usage
-----


Creating parameters
^^^^^^^^^^^^^^^^^^^

There are four types of parameters that can be created: continous, discrete, categorical and constant:

* We can create **continous** parameters with a :attr:`~f3dasm.design.parameter.ContinuousParameter.lower_bound` and :attr:`~f3dasm.design.parameter.ContinuousParameter.upper_bound` with the :class:`~f3dasm.design.parameter.ContinuousParameter` class

.. code-block:: python

  x1 = f3dasm.ContinuousParameter(name='x1', lower_bound=0.0, upper_bound=100.0)
  x2 = f3dasm.ContinuousParameter(name='x2', lower_bound=0.0, upper_bound=4.0)
  y = f3dasm.ContinuousParameter('y') # the default bounds are -np.inf, np.inf
  
* We can create **discrete** parameters with a :attr:`~f3dasm.design.parameter.DiscreteParameter.lower_bound` and :attr:`~f3dasm.design.parameter.DiscreteParameter.upper_bound` with the :class:`~f3dasm.design.parameter.DiscreteParameter` class

.. code-block:: python

  x3 = f3dasm.DiscreteParameter('x3', lower_bound=2, upper_bound=4)
  x4 = f3dasm.DiscreteParameter('x4', lower_bound=74, upper_bound=99)

* We can create **categorical** parameters with a list of strings (:attr:`~f3dasm.design.parameter.CategoricalParameter.categories`) with the :attr:`~f3dasm.design.parameter.CategoricalParameter` class

.. code-block:: python

  x5 = f3dasm.CategoricalParameter('x5', categories=['test1','test2','test3','test4'])
  x6 = f3dasm.CategoricalParameter('x6', categories=['material1','material2','material3'])

* We can create **constant** parameters with any value (:attr:`~f3dasm.design.parameter.ConstantParameter.value`) with the :attr:`~f3dasm.design.parameter.ConstantParameter` class

.. code-block:: python

  x7 = f3dasm.ConstantParameter('x7', value=0.9)


Creating the design space
^^^^^^^^^^^^^^^^^^^^^^^^^

The design space is then constructed by calling the :class:`~f3dasm.design.design.DesignSpace` class and providing:

* a list of input parameters (:attr:`~f3dasm.design.design.DesignSpace.input_space`)
* a list of output parameters (:attr:`~f3dasm.design.design.DesignSpace.output_space`):

.. code-block:: python

  design = f3dasm.DesignSpace(input_space=[x1, x2, x3, x4, x5, x6], output_space=[y])
  
  
Helper function for single-objective, n-dimensional continuous design spaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 
We can make a n-dimensional continous, single-objective design space with the helper function :func:`~f3dasm.base.utils.make_nd_continuous_design`. We have to specify the boundaries for each of the dimensions with a numpy array:

.. code-block:: python

  bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
  design = f3dasm.make_nd_continuous_design(bounds=bounds, dimensions=2)

API Documentation
-----------------

.. automodule:: f3dasm.design.parameter
   :members:
   :noindex:
   :show-inheritance:

.. automodule:: f3dasm.design.design
   :members:
   :noindex:
   :show-inheritance: