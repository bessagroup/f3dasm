Design space
============

The design space can be created with the :class:`~f3dasm.design.design.DesignSpace` class.

Usage
-----

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
