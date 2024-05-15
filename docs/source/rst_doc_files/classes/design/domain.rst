Domain and parameters
=====================

This section will give you information on how to set up your search space with the :ref:`domain <domain>` class and the :ref:`parameters <parameters>`


Domain
------

.. _domain:

The :class:`~f3dasm.design.Domain` is a set of parameter instances that make up the feasible search space.

.. image:: ../../../img/f3dasm-domain.png
    :width: 100%
    :align: center
    :alt: Domain

|


To start, we create an empty domain object:

.. code-block:: python

  from f3dasm.design import Domain

  domain = Domain()


Now we can add some parameters!

.. _parameters:

Input parameters
----------------

Input parameters are singular features of the input search space. They are used to define the search space of the design.

.. image:: ../../../img/f3dasm-parameter.png
   :width: 50%
   :align: center
   :alt: Parameters

|

There are four types of parameters that can be created: :ref:`float <continuous-parameter>`, :ref:`int <discrete-parameter>`, :ref:`categorical <categorical-parameter>` and :ref:`constant <constant-parameter>` parameters.

.. _continuous-parameter:

Floating point parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

* We can create **continous** parameters with a lower bound (:code:`low`) and upper bound (:code:`high`) with the :meth:`~f3dasm.design.Domain.add_float` method:

.. code-block:: python

  domain.add_float(name='x1', low=0.0, high=100.0)
  domain.add_float(name='x2', low=0.0, high=4.0)  

An optional argument :code:`log` can be set to :code:`True` to create a log-scaled parameter:

.. _discrete-parameter:

Discrete parameters
^^^^^^^^^^^^^^^^^^^

* We can create **discrete** parameters with a lower bound (:code:`low`) and upper bound (:code:`high`) with the :meth:`~f3dasm.design.Domain.add_int` method:

.. code-block:: python

  domain.add_int(name='x3', low=2, high=4)
  domain.add_int(name='x4', low=74, high=99)  

An optional argument :code:`step` can be set to an integer value to define the step size between the lower and upper bound. By default the step size is 1.

.. _categorical-parameter:

Categorical parameters
^^^^^^^^^^^^^^^^^^^^^^

* We can create **categorical** parameters with a list of values (:code:`categories`) with the :meth:`~f3dasm.design.Domain.add_category` method:

.. code-block:: python

  domain.add_category(name='x5', categories=['test1','test2','test3','test4'])
  domain.add_category(name='x6', categories=[0.9, 0.2, 0.1, -2])

.. _constant-parameter:

Constant parameters
^^^^^^^^^^^^^^^^^^^

* We can create **constant** parameters with any value (:code:`value`) with the :meth:`~f3dasm.design.Domain.add_constant` method:

.. code-block:: python

  domain.add_constant(name='x7', value=0.9)

.. _domain-from-yaml:

Output parameters
-----------------

Output parameters are the results of evaluating the input design with a data generation model.
Output parameters can hold any type of data, e.g. a scalar value, a vector, a matrix, etc.
Normally, you would not need to define output parameters, as they are created automatically when you store a variable to the :class:`~f3dasm.ExperimentData` object.

.. code-block:: python

  domain.add_output(name='y', to_disk=False)

The :code:`to_disk` argument can be set to :code:`True` to store the output parameter on disk. A reference to the file is stored in the :class:`~f3dasm.ExperimentData` object.
This is useful when the output data is very large, or when the output data is an array-like object.
More information on storing output can be found in :ref:`this section <storing-output-experiment-sample>`

Domain from a `hydra <https://hydra.cc/>`_ configuration file
-------------------------------------------------------------

If you are using `hydra <https://hydra.cc/>`_ to manage your configuration files, you can create a domain from a configuration file. 
Your config needs to have a key (e.g. :code:`domain`) that has a dictionary with the parameter names (e.g. :code:`param_1`) as keys 
and a dictionary with the parameter type (:code:`type`) and the corresponding arguments as values:

.. code-block:: yaml
   :caption: config.yaml

    domain:
        param_1:
            type: float
            low: -1.0
            high: 1.0
        param_2:
            type: int
            low: 1
            high: 10
        param_3:
            type: category
            categories: ['red', 'blue', 'green', 'yellow', 'purple']
        param_4:
            type: constant
            value: some_value

  
The domain can now be created by calling the :func:`~f3dasm.design.Domain.from_yaml` method:

.. code-block:: python

    import hydra

    @hydra.main(config_path="conf", config_name="config")
    def my_app(cfg):
      domain = Domain.from_yaml(cfg.domain)

Helper function for single-objective, n-dimensional continuous domains
----------------------------------------------------------------------
 
We can make easily make a :math:`n`-dimensional continous domain with the helper function :func:`~f3dasm.design.make_nd_continuous_domain`. 
We have to specify the boundaries (``bounds``) for each of the dimensions with a list of lists or numpy :class:`~numpy.ndarray`:

.. code-block:: python
  
  from f3dasm.design import make_nd_continuous_domain
  import numpy as np
  bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
  domain = make_nd_continuous_domain(bounds=bounds, dimensionality=2)


.. minigallery:: f3dasm.design.Domain
    :add-heading: Examples using the `Domain` object
    :heading-level: -
