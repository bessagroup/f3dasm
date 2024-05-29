"""
Introduction to domain and parameters
=====================================

This section will give you information on how to set up your search space with the :class:`~f3dasm.design.Domain` class and the paramaters
The :class:`~f3dasm.design.Domain` is a set of parameter instances that make up the feasible search space.
"""
###############################################################################
# To start, we create an empty domain object:
import numpy as np
from hydra import compose, initialize

from f3dasm.design import Domain, make_nd_continuous_domain

domain = Domain()

###############################################################################
# Input parameters
# ----------------
#
# Now we well add some input parameters:
# There are four types of parameters that can be created:
#
# - floating point parameters

domain.add_float(name='x1', low=0.0, high=100.0)
domain.add_float(name='x2', low=0.0, high=4.0)

###############################################################################
# - discrete integer parameters

domain.add_int(name='x3', low=2, high=4)
domain.add_int(name='x4', low=74, high=99)

###############################################################################
# - categorical parameters

domain.add_category(name='x5', categories=['test1', 'test2', 'test3', 'test4'])
domain.add_category(name='x6', categories=[0.9, 0.2, 0.1, -2])

###############################################################################
# - constant parameters

domain.add_constant(name='x7', value=0.9)

###############################################################################
# We can print the domain object to see the parameters that have been added:

print(domain)

###############################################################################
# Output parameters
# -----------------
#
# Output parameters are the results of evaluating the input design with a data generation model.
# Output parameters can hold any type of data, e.g. a scalar value, a vector, a matrix, etc.
# Normally, you would not need to define output parameters, as they are created automatically when you store a variable to the :class:`~f3dasm.ExperimentData` object.

domain.add_output(name='y', to_disk=False)

###############################################################################
# The :code:`to_disk` argument can be set to :code:`True` to store the output parameter on disk. A reference to the file is stored in the :class:`~f3dasm.ExperimentData` object.
# This is useful when the output data is very large, or when the output data is an array-like object.

###############################################################################
# Filtering the domain
# --------------------
#
# The domain object can be filtered to only include certain types of parameters.
# This might be useful when you want to create a design of experiments with only continuous parameters, for example.
# The attributes :attr:`~f3dasm.design.Domain.continuous`, :attr:`~f3dasm.design.Domain.discrete`, :attr:`~f3dasm.design.Domain.categorical`, and :attr:`~f3dasm.design.Domain.constant` can be used to filter the domain object.

print(f"Continuous domain: {domain.continuous}")
print(f"Discrete domain: {domain.discrete}")
print(f"Categorical domain: {domain.categorical}")
print(f"Constant domain: {domain.constant}")

###############################################################################
# Helper function for single-objective, n-dimensional continuous domains
# ----------------------------------------------------------------------
#
# We can make easily make a :math:`n`-dimensional continous domain with the helper function :func:`~f3dasm.design.make_nd_continuous_domain`.
# We have to specify the boundaries (``bounds``) for each of the dimensions with a list of lists or numpy :class:`~numpy.ndarray`:

bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
domain = make_nd_continuous_domain(bounds=bounds)

print(domain)
