Parameters
==========

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

Implemented parameters
----------------------

======================== ======================================================================
Name                      Docs of the Python class                                             
======================== ======================================================================
CategoricalParameter     :class:`~f3dasm.design.parameter.CategoricalParameter`                  
ContinuousParameter      :class:`~f3dasm.design.parameter.ContinuousParameter`                  
DiscreteParameter        :class:`~f3dasm.design.parameter.DiscreteParameter`                     
ConstantParameter        :class:`~f3dasm.design.parameter.ConstantParameter`                    
======================== ======================================================================