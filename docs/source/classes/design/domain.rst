Domain
======

The :class:`~f3dasm.design.domain.Domain` is a set of :class:`f3dasm.design.parameter.Parameter` instances that make up the feasible search space.

.. image:: ../../img/f3dasm-domain.png
    :width: 100%
    :align: center
    :alt: Domain

|

Creating the Domain
-------------------

From a dictionary
^^^^^^^^^^^^^^^^^

The domain can be constructed by initializing the :class:`~f3dasm.design.domain.Domain` class and 
providing an attribute (:attr:`~f3dasm.design.domain.Domain.input_space`) containing string names as keys and parameters as values.

.. code-block:: python

  from f3dasm import Domain, ContinuousParameter, DiscreteParameter, CategoricalParameter, ConstantParameter

  param_1 = f3dasm.ContinuousParameter(lower_bound=-1.0, upper_bound=1.0)
  param_2 = f3dasm.DiscreteParameter(lower_bound=1, upper_bound=10)
  param_3 = f3dasm.CategoricalParameter(categories=['red', 'blue', 'green', 'yellow', 'purple'])
  param_4 = f3dasm.ConstantParameter(value='some_value')

  domain = f3dasm.Domain(input_space={'param_1': param_1, 'param_2': param_2, 'param_3': param_3, 'param_4': param_4})

From a dataframe
^^^^^^^^^^^^^^^^

The domain can also be infered from a pandas dataframe containg samples. 
The dataframe needs to have the column names as the parameter names and the values as the parameter values. The dataframe can contain any number of samples. The domain will be infered from the first sample.

.. code-block:: python

  import pandas as pd
  from f3dasm import Domain

  df = pd.DataFrame({'param_1': [0.1, -0.3, 0.6], 'param_2': [1, 3, 9], 'param_3': ['red', 'blue', 'purple'], 'param_4': ['some_value', 'some_value', 'some_value']})
  domain = Domain.from_dataframe(df)

.. note:: 
  
  Constructing the dataframe by inferring it from samples can be useful if you have a large number of parameters and you don't want to manually specify the domain.
  However, the domain will be a guess based on the information it has and this might be inacurate. 
  In the above example, the domain for param_3 will not include the 'green' and 'yellow' categories, as they do not appear in the samples.


From a hydra configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are using hydra to manage your configuration files, you can create a domain from a configuration file. Your config needs to have the following structure:

.. code-block:: yaml
   :caption: config.yaml

    domain:
        input_space:
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

  
The same domain can now be created by calling the :func:`~f3dasm.design.domain.Domain.from_yaml` method:

.. code-block:: python

    import hydra

    @hydra.main(config_path="conf", config_name="config")
    def my_app(cfg):
      domain = Domain.from_yaml(cfg.domain)


Storing a domain
----------------

You can store a domain to disk by calling the :func:`~f3dasm.design.domain.Domain.store` method:

.. code-block:: python

  domain.store('my_domain')

This will store the domain as a pickle file. It can be loaded into memory again by calling the :func:`~f3dasm.design.domain.Domain.from_file` method:

.. code-block:: python

  domain = Domain.from_file('my_domain')

Helper function for single-objective, n-dimensional continuous Domains
----------------------------------------------------------------------
 
We can make easily make a :math:`n`-dimensional continous domain with the helper function :func:`~f3dasm.design.domain.make_nd_continuous_domain`. We have to specify the boundaries for each of the dimensions with a numpy array:

.. code-block:: python

  bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
  design = f3dasm.make_nd_continuous_domain(bounds=bounds, dimensionality=2)
