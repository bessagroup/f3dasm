Domain and parameters
=====================

This section will give you information on how to set up your search space with the :ref:`domain <domain>` class and the :ref:`parameter classes <parameters>`

.. _parameters:

Parameters
----------

Parameters are singular features of the input search space. They are used to define the search space of the design.

.. image:: ../../../img/f3dasm-parameter.png
   :width: 50%
   :align: center
   :alt: Parameters

|

There are four types of parameters that can be created: :class:`~f3dasm.design.ContinuousParameter`, :class:`~f3dasm.design.DiscreteParameter`, :class:`~f3dasm.design.CategoricalParameter` and :class:`~f3dasm.design.ConstantParameter`:

Continuous Parameter
^^^^^^^^^^^^^^^^^^^^

* We can create **continous** parameters with a :attr:`~f3dasm.design.ContinuousParameter.lower_bound` and :attr:`~f3dasm.design.ContinuousParameter.upper_bound` with the :class:`~f3dasm.design.ContinuousParameter` class

.. code-block:: python

  x1 = f3dasm.ContinuousParameter(lower_bound=0.0, upper_bound=100.0)
  x2 = f3dasm.ContinuousParameter(lower_bound=0.0, upper_bound=4.0)

Discrete Parameter
^^^^^^^^^^^^^^^^^^

* We can create **discrete** parameters with a :attr:`~f3dasm.design.DiscreteParameter.lower_bound` and :attr:`~f3dasm.design.DiscreteParameter.upper_bound` with the :class:`~f3dasm.design.DiscreteParameter` class

.. code-block:: python

  x3 = f3dasm.DiscreteParameter(lower_bound=2, upper_bound=4)
  x4 = f3dasm.DiscreteParameter(lower_bound=74, upper_bound=99)

Categorical Parameter
^^^^^^^^^^^^^^^^^^^^^

* We can create **categorical** parameters with a list of items (:attr:`~f3dasm.design.CategoricalParameter.categories`) with the :class:`~f3dasm.design.CategoricalParameter` class

.. code-block:: python

  x5 = f3dasm.CategoricalParameter(categories=['test1','test2','test3','test4'])
  x6 = f3dasm.CategoricalParameter(categories=[0.9, 0.2, 0.1, -2])

Constant Parameter
^^^^^^^^^^^^^^^^^^

* We can create **constant** parameters with any value (:attr:`~f3dasm.design.ConstantParameter.value`) with the :class:`~f3dasm.design.ConstantParameter` class

.. code-block:: python

  x7 = f3dasm.ConstantParameter(value=0.9)


Domain
------

.. _domain:

The :class:`~f3dasm.design.Domain` is a set of :class:`f3dasm.design.Parameter` instances that make up the feasible search space.

.. image:: ../../../img/f3dasm-domain.png
    :width: 100%
    :align: center
    :alt: Domain

|

Domain from a dictionary
^^^^^^^^^^^^^^^^^^^^^^^^

The domain can be constructed by initializing the :class:`~f3dasm.design.Domain` class and 
providing an attribute (:attr:`~f3dasm.design.Domain.input_space`) containing string names as keys and parameters as values.

.. code-block:: python

  from f3dasm import Domain, ContinuousParameter, DiscreteParameter, CategoricalParameter, ConstantParameter

  param_1 = f3dasm.ContinuousParameter(lower_bound=-1.0, upper_bound=1.0)
  param_2 = f3dasm.DiscreteParameter(lower_bound=1, upper_bound=10)
  param_3 = f3dasm.CategoricalParameter(categories=['red', 'blue', 'green', 'yellow', 'purple'])
  param_4 = f3dasm.ConstantParameter(value='some_value')

  domain = f3dasm.Domain(input_space={'param_1': param_1, 'param_2': param_2, 'param_3': param_3, 'param_4': param_4})

.. _domain-from-yaml:

Domain from a `hydra <https://hydra.cc/>`_ configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are using `hydra <https://hydra.cc/>`_ to manage your configuration files, you can create a domain from a configuration file. Your config needs to have the following key:

.. code-block:: yaml
   :caption: config.yaml

    domain:
        param_1:
            _target_: f3dasm.ContinuousParameter
            lower_bound: -1.0
            upper_bound: 1.0
        param_2:
            _target_: f3dasm.DiscreteParameter
            lower_bound: 1
            upper_bound: 10
        param_3:
            _target_: f3dasm.CategoricalParameter
            categories: ['red', 'blue', 'green', 'yellow', 'purple']
        param_4:
            _target_: f3dasm.ConstantParameter
            value: some_value

  
The same domain can now be created by calling the :func:`~f3dasm.design.Domain.from_yaml` method:

.. code-block:: python

    import hydra

    @hydra.main(config_path="conf", config_name="config")
    def my_app(cfg):
      domain = Domain.from_yaml(cfg.domain)

Helper function for single-objective, n-dimensional continuous Domains
----------------------------------------------------------------------
 
We can make easily make a :math:`n`-dimensional continous domain with the helper function :func:`~f3dasm.design.make_nd_continuous_domain`. 
We have to specify the boundaries (``bounds``) for each of the dimensions with a list of lists or numpy :class:`~numpy.ndarray`:

.. code-block:: python

  bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
  domain = f3dasm.make_nd_continuous_domain(bounds=bounds, dimensionality=2)
