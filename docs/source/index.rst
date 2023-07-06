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
   :caption: 1. Design of Experiments
   :hidden:
   :glob:

   classes/design/parameters
   classes/design/domain
   classes/design/experimentdata
   classes/design/design
   classes/sampling/sampling
   classes/design/apidoc

.. toctree::
   :maxdepth: 2
   :caption: 2. Data Generation
   :hidden:
   :glob:

   classes/datageneration/datagenerator
   classes/datageneration/functions
   classes/datageneration/apidoc


.. toctree::
   :maxdepth: 2
   :caption: 3. Machine Learning
   :hidden:
   :glob:

   classes/machinelearning/apidoc

.. toctree::
   :maxdepth: 2
   :caption: 4. Optimization
   :hidden:
   :glob:

   classes/optimization/optimizers
   classes/optimization/apidoc

.. toctree::
   :maxdepth: 2
   :caption: Workflow execution
   :hidden:
   :glob:

   classes/workflow/workflow
   classes/workflow/cluster
   classes/workflow/hydra

.. include:: readme.rst


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
