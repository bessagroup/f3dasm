
Welcome to F3DASM's documentation!
==================================

.. |f3dasm| replace:: F3DASM
.. _f3dasm: https://github.com/bessagroup/F3DASM>

.. |abaqus| replace:: Abaqus
.. _abaqus: https://www.3ds.com/products-services/simulia/products/abaqus/


|f3dasm|_, which stands for **Framework for Data-Driven Design and Analysis of Structures**, is an open-source Python library under the BSD license that intends to facilitate the design of new materials and structures following a data-driven approach.

For now, the library uses |abaqus| to perform numerical simulations (a short-term goal is to also use an open-source
FEM software).

All the user has to provide is a parameterized script to create and post-process a numerical model. |f3dasm|_ is responsable to manage the design of experiments (interfacing with `SALib <https://salib.readthedocs.io/en/latest/>`_), run the different datapoints (possibly in parallel) and collect the data into a `Pandas <https://pandas.pydata.org/>`_ dataframe (very ameanable for the application
of Machine Learning using e.g. `scikit-learn <https://scikit-learn.org/stable/>`_ or tensorflow's `keras <https://www.tensorflow.org/api_docs/python/tf/keras>`_).

Additionally, |f3dasm|_ offers utils for fast and easy development of new |abaqus|_ models (e.g. micromechanical models where the implementation of the boundary conditions is responsability of |f3dasm|_).


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   references
   basic_supercompressible


.. toctree::
   :maxdepth: 2
   :caption: Examples



.. toctree::
   :maxdepth: 2
   :caption: Advanced users




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
