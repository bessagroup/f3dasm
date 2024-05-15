"""
Use the built-in benchmark functions
====================================

In this example, we will use the built-in benchmark functions provided by :mod:`f3dasm.datageneration.functions` to generate output for a data-driven experiment.
"""

import matplotlib.pyplot as plt

from f3dasm import ExperimentData
from f3dasm.design import make_nd_continuous_domain

###############################################################################
# :mod:`f3dasm` ships with a set of benchmark functions that can be used to test the performance of
# optimization algorithms or to mock some expensive simulation in order to test the data-driven process.
# These benchmark functions are taken and modified from the `Python Benchmark Test Optimization Function Single Objective <https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective>`_ github repository.
#
# Let's start by creating a continuous domain
# with 2 input variables, each ranging from -1.0 to 1.0

domain = make_nd_continuous_domain([[-1., 1.], [-1., 1.]])

###############################################################################
# We generate the input data by sampling the domain equally spaced with the grid sampler and create the :class:`~f3dasm.ExperimentData` object:

experiment_data = ExperimentData.from_sampling(
    'grid', domain=domain, stepsize_continuous_parameters=0.1)

print(experiment_data)

###############################################################################
# Evaluating a 2D version of the Ackley function is as simple as
# calling the :meth:`~f3dasm.ExperimentData.evaluate` method with the function name as the ``data_generator`` argument.
#
# In addition, you can provide a dictionary (``kwargs``) with the followinging keywords to the :class:`~f3dasm.design.ExperimentData.evaluate` method:
#
# * ``scale_bounds``: A 2D list of floats that define the scaling lower and upper boundaries for each dimension. The normal benchmark function box-constraints will be scaled to these boundaries.
# * ``noise``: A float that defines the standard deviation of the Gaussian noise that is added to the objective value.
# * ``offset``: A boolean value. If ``True``, the benchmark function will be offset by a constant vector that will be randomly generated [1]_.
# * ``seed``: Seed for the random number generator for the ``noise`` and ``offset`` calculations.
#
# .. [1] As benchmark functions usually have their minimum at the origin, the offset is used to test the robustness of the optimization algorithm.

experiment_data.evaluate(data_generator='Ackley', kwargs={
                         'scale_bounds': domain.get_bounds(), 'offset': False})

###############################################################################
# The function values are stored in the ``y`` variable of the output data:

print(experiment_data)

###############################################################################

arr_in, arr_out = experiment_data.to_numpy()
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.scatter(arr_in[:, 0], arr_in[:, 1], arr_out.ravel())
_ = ax.set_xlabel('$x_0$')
_ = ax.set_ylabel('$x_1$')
_ = ax.set_zlabel('$f(x)$')

###############################################################################
# A complete list of all the implemented benchmark functions can be found :ref:`here <implemented-benchmark-functions>`
