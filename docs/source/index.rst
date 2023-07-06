.. sample documentation master file, created by
   sphinx-quickstart on Mon Apr 16 21:22:43 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

f3dasm
======

.. toctree::
   :maxdepth: 3
   :caption: General
   :hidden:
   :glob:
       
   general/gettingstarted
   general/overview

.. toctree::
   :maxdepth: 2
   :caption: Design & Sampling
   :hidden:
   :glob:

   classes/design/parameters
   classes/design/design
   classes/design/experimentdata
   classes/sampling/sampling
   classes/design/apidoc

.. toctree::
   :maxdepth: 2
   :caption: Data Generation
   :hidden:
   :glob:

   classes/datageneration/datagenerator
   classes/datageneration/functions
   classes/datageneration/apidoc


.. toctree::
   :maxdepth: 2
   :caption: Machine Learning
   :hidden:
   :glob:

   classes/machinelearning/apidoc

.. toctree::
   :maxdepth: 2
   :caption: Optimization
   :hidden:
   :glob:

   classes/optimization/optimizers
   classes/optimization/apidoc

.. toctree::
   :maxdepth: 2
   :caption: Workflow execution
   :hidden:
   :glob:

   classes/workflow/parallelization
   classes/workflow/hydra
   classes/workflow/apidoc

.. include:: readme.rst


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
