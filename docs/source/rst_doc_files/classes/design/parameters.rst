Parameters
==========

Parameters are the variables that we want to  .. parameterize!. 

.. image:: ../../../img/f3dasm-parameter.png
   :width: 50%
   :align: center
   :alt: Parameters

There are four types of parameters that can be created: :class:`~f3dasm.design.parameter.ContinuousParameter`, :class:`~f3dasm.design.parameter.DiscreteParameter`, :class:`~f3dasm.design.parameter.CategoricalParameter` and :class:`~f3dasm.design.parameter.ConstantParameter`:

Continuous Parameter
--------------------

* We can create **continous** parameters with a :attr:`~f3dasm.design.parameter.ContinuousParameter.lower_bound` and :attr:`~f3dasm.design.parameter.ContinuousParameter.upper_bound` with the :class:`~f3dasm.design.parameter.ContinuousParameter` class

.. code-block:: python

  x1 = f3dasm.ContinuousParameter(lower_bound=0.0, upper_bound=100.0)
  x2 = f3dasm.ContinuousParameter(lower_bound=0.0, upper_bound=4.0)

Discrete Parameter
------------------

* We can create **discrete** parameters with a :attr:`~f3dasm.design.parameter.DiscreteParameter.lower_bound` and :attr:`~f3dasm.design.parameter.DiscreteParameter.upper_bound` with the :class:`~f3dasm.design.parameter.DiscreteParameter` class

.. code-block:: python

  x3 = f3dasm.DiscreteParameter(lower_bound=2, upper_bound=4)
  x4 = f3dasm.DiscreteParameter(lower_bound=74, upper_bound=99)

Categorical Parameter
---------------------

* We can create **categorical** parameters with a list of items (:attr:`~f3dasm.design.parameter.CategoricalParameter.categories`) with the :class:`~f3dasm.design.parameter.CategoricalParameter` class

.. code-block:: python

  x5 = f3dasm.CategoricalParameter(categories=['test1','test2','test3','test4'])
  x6 = f3dasm.CategoricalParameter(categories=[0.9, 0.2, 0.1, -2])

Constant Parameter
---------------------

* We can create **constant** parameters with any value (:attr:`~f3dasm.design.parameter.ConstantParameter.value`) with the :class:`~f3dasm.design.parameter.ConstantParameter` class

.. code-block:: python

  x7 = f3dasm.ConstantParameter(value=0.9)