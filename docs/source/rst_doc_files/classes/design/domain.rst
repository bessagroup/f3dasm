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


To start, we instantiate an empty domain object:

.. code-block:: python

  from f3dasm.design import Domain

  domain = Domain()


Now we can gradually add some parameters!

.. _parameters:

Parameters
----------

Parameters are singular features of the input search space. They are used to define the search space of the design.

.. image:: ../../../img/f3dasm-parameter.png
   :width: 50%
   :align: center
   :alt: Parameters

|

There are four types of parameters that can be created: :ref:`float <continuous-parameter>`, :ref:`int <discrete-parameter>`, :ref:`categorical <categorical-parameter>` and :ref:`constant <constant-parameter>` parameters.

.. _continuous-parameter:

Floating point parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

* We can create **continous** parameters with a :code:`low` and :code:`high` boundary with the :meth:`~f3dasm.design.Domain.add_float` method:

.. code-block:: python

  domain.add_float(name='x1', low=0.0, high=100.0)
  domain.add_float(name='x2', low=0.0, high=4.0)  

.. _discrete-parameter:

Discrete parameters
^^^^^^^^^^^^^^^^^^^

* We can create **discrete** parameters with a :code:`low` and :code:`high` boundary with the :meth:`~f3dasm.design.Domain.add_int` method:

.. code-block:: python

  domain.add_int(name='x3', low=2, high=4)
  domain.add_int(name='x4', low=74, high=99)  

.. _categorical-parameter:

Categorical parameters
^^^^^^^^^^^^^^^^^^^^^^

* We can create **categorical** parameters with a list of items (:code:`categories`) with the :meth:`~f3dasm.design.Domain.add_category` method:

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

Helper function for single-objective, n-dimensional continuous Domains
----------------------------------------------------------------------
 
We can make easily make a :math:`n`-dimensional continous domain with the helper function :func:`~f3dasm.design.make_nd_continuous_domain`. 
We have to specify the boundaries (``bounds``) for each of the dimensions with a list of lists or numpy :class:`~numpy.ndarray`:

.. code-block:: python

  bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
  domain = f3dasm.make_nd_continuous_domain(bounds=bounds, dimensionality=2)
