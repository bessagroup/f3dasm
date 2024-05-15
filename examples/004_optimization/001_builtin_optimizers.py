"""
Use the built-in optimization algorithms
========================================

In this example, we will use the built-in optimization algorithms provided by the :mod:`f3dasm.optimization` submodule to optimize the Rosenbrock benchmark function.
"""

import matplotlib.pyplot as plt

from f3dasm import ExperimentData
from f3dasm.design import make_nd_continuous_domain
from f3dasm.optimization import OPTIMIZERS

###############################################################################
# We create a 3D continous domain and sample one point from it.

domain = make_nd_continuous_domain([[-1., 1.], [-1., 1.], [-1., 1.]])

experimentdata = ExperimentData.from_sampling(
    domain=domain, sampler="random", seed=42, n_samples=1)

print(experimentdata)

###############################################################################
# We evaluate the sample point on the Rosenbrock benchmark function:

experimentdata.evaluate(data_generator='Rosenbrock', kwargs={
                        'scale_bounds': domain.get_bounds(), 'offset': False})

print(experimentdata)

###############################################################################
# We call the :meth:`~f3dasm.ExperimentData.optimize` method with ``optimizer='CG'``
# and ``data_generator='Rosenbrock'`` to optimize the Rosenbrock benchmark function with the
# Conjugate Gradient Optimizer:

experimentdata.optimize(optimizer='CG', data_generator='Rosenbrock', kwargs={
                        'scale_bounds': domain.get_bounds(), 'offset': False},
                        iterations=50)

print(experimentdata)

###############################################################################
# We plot the convergence of the optimization process:

_, df_output = experimentdata.to_pandas()

fig, ax = plt.subplots()
ax.plot(df_output)
_ = ax.set_xlabel('number of function evaluations')
_ = ax.set_ylabel('$f(x)$')
ax.set_yscale('log')

###############################################################################
# Hyper-parameters of the optimizer can be passed as dictionary to the :meth:`~f3dasm.ExperimentData.optimize` method.
# If none are provided, default hyper-parameters are used. The hyper-parameters are specific to the optimizer used, and can be found in the corresponding documentation.
#
# An overview of the available optimizers can be found in :ref:`this section <implemented optimizers>` of the documentation
# Access to more off-the-shelf optimizers requires the installation of the `f3dasm_optimize <https://bessagroup.github.io/f3dasm_optimize/>`_ package and its corresponding dependencies.
# You can check which optimizers can be used by inspecting the ``f3dasm.optimization.OPTIMIZERS`` variable:

print(OPTIMIZERS)
