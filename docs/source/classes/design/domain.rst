Domain
============

The domain can be created with the :class:`~f3dasm.design.domain.Domain` class.

Usage
-----

Creating the Domain
^^^^^^^^^^^^^^^^^^^^^^^^^

The domain is then constructed by calling the :class:`~f3dasm.design.domain.Domain` class and providing:

* a dictionary with names as keys and output parameters as values (:attr:`~f3dasm.design.domain.Domain.input_space`)
* a dictionary with names as keys and output parameters as values (:attr:`~f3dasm.design.domain.Domain.output_space`):

.. code-block:: python

  design = f3dasm.Domain(input_space=[x1, x2, x3, x4, x5, x6], output_space=[y])
  
  
Helper function for single-objective, n-dimensional continuous Domains
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 
We can make a n-dimensional continous, single-objective domain with the helper function :func:`~f3dasm.design.domain.make_nd_continuous_domain`. We have to specify the boundaries for each of the dimensions with a numpy array:

.. code-block:: python

  bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
  design = f3dasm.make_nd_continuous_domain(bounds=bounds, dimensions=2)
