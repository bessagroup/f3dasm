"""
Use the built-in sampling strategies
====================================

In this example, we will use the built-in sampling strategies provided by :mod:`f3dasm` to generate samples for a data-driven experiment.
"""

from matplotlib import pyplot as plt

from f3dasm import ExperimentData
from f3dasm.design import make_nd_continuous_domain

###############################################################################
# We create 2D continuous input domain with the :func:`~f3dasm.design.make_nd_continuous_domain` helper function:

domain = make_nd_continuous_domain(bounds=[[0., 1.], [0., 1.]])
print(domain)

###############################################################################
# You can create an :class:`~f3dasm.ExperimentData` object with the :meth:`~f3dasm.ExperimentData.from_sampling` constructor directly:

data_random = ExperimentData.from_sampling(
    domain=domain, n_samples=10, sampler='random', seed=42)

fig, ax = plt.subplots(figsize=(4, 4))

print(data_random)

df_random, _ = data_random.to_pandas()
ax.scatter(df_random.iloc[:, 0], df_random.iloc[:, 1])
ax.set_xlabel(domain.names[0])
ax.set_ylabel(domain.names[1])

###############################################################################
# :mod:`f3dasm` provides several built-in samplers.
# The example below shows how to use the Latin Hypercube Sampling (LHS) sampler:

data_lhs = ExperimentData.from_sampling(
    domain=domain, n_samples=10, sampler='latin', seed=42)

fig, ax = plt.subplots(figsize=(4, 4))

print(data_lhs)

df_lhs, _ = data_lhs.to_pandas()
ax.scatter(df_lhs.iloc[:, 0], df_lhs.iloc[:, 1])
ax.set_xlabel(domain.names[0])
ax.set_ylabel(domain.names[1])

###############################################################################
# More information all the available samplers can be found in :ref:`here <implemented samplers>`.
