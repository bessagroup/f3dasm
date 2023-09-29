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
       
   rst_doc_files/general/gettingstarted
   rst_doc_files/general/overview

.. toctree::
   :maxdepth: 2
   :caption: Design of Experiments
   :hidden:
   :glob:

   rst_doc_files/classes/design/parameters
   rst_doc_files/classes/design/domain
   rst_doc_files/classes/sampling/sampling

.. toctree::
   :maxdepth: 2
   :caption: Data
   :hidden:
   :glob:

   rst_doc_files/classes/design/experimentdata
   rst_doc_files/classes/design/experimentsample

.. toctree::
   :maxdepth: 2
   :caption: Data Generation
   :hidden:
   :glob:

   rst_doc_files/classes/datageneration/datagenerator
   rst_doc_files/classes/datageneration/functions
   rst_doc_files/classes/datageneration/f3dasm-simulate


.. toctree::
   :maxdepth: 2
   :caption: Machine Learning
   :hidden:
   :glob:

   rst_doc_files/classes/machinelearning/machinelearning

.. toctree::
   :maxdepth: 2
   :caption: Optimization
   :hidden:
   :glob:

   rst_doc_files/classes/optimization/optimizers
   rst_doc_files/classes/optimization/f3dasm-optimize

.. toctree::
   :maxdepth: 2
   :caption: Workflow execution
   :hidden:
   :glob:

   rst_doc_files/classes/workflow/workflow
   rst_doc_files/classes/workflow/hydra
   rst_doc_files/classes/workflow/cluster

.. toctree::
   :name: apitoc
   :caption: API
   :hidden:

   rst_doc_files/reference/index.rst
   Code <_autosummary/f3dasm>

.. include:: readme.rst