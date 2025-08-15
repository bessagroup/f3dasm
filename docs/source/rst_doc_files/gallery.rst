.. _tutorials:

Tutorials
=========

Design-of-experiments
---------------------

The submodule :mod:`f3dasm.design` contains the  :class:`~f3dasm.design.Domain` object that makes up your feasible search space.

.. nblinkgallery::
    :name: designofexperiments
    :glob:

    ../notebooks/design/*

Managing experiments with the ExperimentData object
---------------------------------------------------

The :class:`~f3dasm.ExperimentData` object is the main object used to store implementations of a design-of-experiments, 
keep track of results, perform optimization and extract data for machine learning purposes.

All other processses of :mod:`f3dasm` use this object to manipulate and access data about your experiments.

.. nblinkgallery::
    :name: experimentdata
    :glob:

    ../notebooks/experimentdata/*

Designing your data-driven process
----------------------------------

Use the :class:`~f3dasm.Block` abstraction to manipulate the :class:`~f3dasm.ExperimentData` object and create your data-driven process.

.. nblinkgallery::
    :name: datadriven
    :glob:

    ../notebooks/data-driven/*

Built-in implementations
------------------------

The :mod:`f3dasm` package comes with built-in implementations of the :class:`~f3dasm.Block` object that you can use to create your data-driven process.

.. nblinkgallery::
    :name: builtins
    :glob:

    ../notebooks/builtins/*

Integration with 3rd party libraries
------------------------------------

The :mod:`f3dasm` is mend to be integrated with 3rd party libraries to extend its functionality.

.. nblinkgallery::
    :name: integration
    :glob:

    ../notebooks/integration/*

Integration with hydra
----------------------

Examples that integrate the :mod:`f3dasm` package with the configuration manager hydra

.. nblinkgallery::
    :name: hydra
    :glob:

    ../notebooks/hydra/*

