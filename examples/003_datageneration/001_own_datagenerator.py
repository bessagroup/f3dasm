"""
Implement your own datagenerator: car stopping distance problem
===============================================================

In this example, we will implement a custom data generator that generates output for a data-driven experiment.
We will use the 'car stopping distance' problem as an example.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from f3dasm import ExperimentData
from f3dasm.datageneration import DataGenerator
from f3dasm.design import Domain

###############################################################################
#
# Car stopping distance problem
# -----------------------------
#
# .. image:: ../../img/reaction-braking-stopping.png
#    :width: 70%
#    :align: center
#    :alt: Workflow
#
# Car stopping distance :math:`y` as a function of its velocity :math:`x` before it starts braking:
#
# .. math::
#
#     y = z x + \frac{1}{2 \mu g} x^2 = z x + 0.1 x^2
#
#
# - :math:`z` is the driver's reaction time (in seconds)
# - :math:`\mu` is the road/tires coefficient of friction (we assume :math:`\mu=0.5`)
# - :math:`g` is the acceleration of gravity (assume :math:`g=10 m/s^2`).
#
# .. math::
#
#     y = d_r + d_{b}
#
# where :math:`d_r` is the reaction distance, and :math:`d_b` is the braking distance.
#
# Reaction distance :math:`d_r`
#
# .. math::
#
#     d_r = z x
#
# with :math:`z` being the driver's reaction time, and :math:`x` being the velocity of the car at the start of braking.
#
# Kinetic energy of moving car:
#
# .. math::
#
#     E = \frac{1}{2}m x^2
#
# where :math:`m` is the car mass.
#
# Work done by braking:
#
# .. math::
#
#     W = \mu m g d_b
#
#
# where :math:`\mu` is the coefficient of friction between the road and the tire, :math:`g` is the acceleration of gravity, and :math:`d_b` is the car braking distance.
#
# The braking distance follows from :math:`E=W`:
#
# .. math::
#
#     d_b = \frac{1}{2\mu g}x^2
#
# Therefore, if we add the reacting distance :math:`d_r` to the braking distance :math:`d_b` we get the stopping distance :math:`y`:
#
# .. math::
#
#     y = d_r + d_b = z x + \frac{1}{2\mu g} x^2
#
#
# Every driver has its own reaction time :math:`z`
# Assume the distribution associated to :math:`z` is Gaussian with mean :math:`\mu_z=1.5` seconds and variance :math:`\sigma_z^2=0.5^2`` seconds\ :sup:`2`:
#
# .. math::
#
#     z \sim \mathcal{N}(\mu_z=1.5,\sigma_z^2=0.5^2)
#
#
# We create a function that generates the stopping distance :math:`y` given the velocity :math:`x` and the reaction time :math:`z`:


def y(x):
    z = norm.rvs(1.5, 0.5, size=1)
    y = z*x + 0.1*x**2
    return y


###############################################################################
# Implementing the data generator
# -------------------------------
#
# Implementing this relationship in :mod:`f3dasm` can be done in two ways:
#
#
# 1. Directly using a function
# 2. Providing an object from a custom class that inherits from the :class:`~f3dasm.datageneration.DataGenerator` class.
#
# Using a function directly
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We can use the function :func:`y(x)` directly as the data generator. We will demonstrate this in the following example code:
#
#
# In order to create an :class:`~f3dasm.ExperimentData` object, we have to first create a domain
domain = Domain()
domain.add_float('x', low=0., high=100.)

###############################################################################
# For demonstration purposes, we will generate a dataset of stopping distances for velocities between 3 and 83 m/s.

N = 33  # number of points to generate
Data_x = np.linspace(3, 83, 100)

###############################################################################
# We can construct an :class:`~f3dasm.ExperimentData` object with the :class:`~f3dasm.design.Domain` and the numpy array:

experiment_data = ExperimentData(input_data=Data_x, domain=domain)
print(experiment_data)

###############################################################################
# As you can see, the ExperimentData object has been created successfully and the jobs have the label 'open'.
# This means that the output has not been generated yet. We can now compute the stopping distance by calling the :meth:`~f3dasm.ExperimentData.evaluate` method:
# We have to provide the function as the ``data_generator`` argument and provide name of the return value as the ``output_names`` argument:

experiment_data.evaluate(data_generator=y, output_names=['y'])

arr_in, arr_out = experiment_data.to_numpy()

fig, ax = plt.subplots()
ax.scatter(arr_in, arr_out.flatten(), s=2)
_ = ax.set_xlabel('Car velocity ($m/s$)')
_ = ax.set_ylabel('Stopping distance ($m$)')

###############################################################################
# The experiments have been evaluated and the jobs value has been set to 'finished'

print(experiment_data)

###############################################################################
#
# Using the DataGenerator class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We can also implement the data generator as a class that inherits from the :class:`~f3dasm.datageneration.DataGenerator` class.
# This allows for more flexibility and control over the data generation process.

experiment_data_class = ExperimentData(input_data=Data_x, domain=domain)

###############################################################################
# The custom data generator class should have an :meth:`~f3dasm.datageneration.DataGenerator.execute` method.
# In this method, we can access the experiment using the :attr:`~f3dasm.datageneration.DataGenerator.experiment_sample` attribute.
# We can store the output of the data generation process using the :meth:`~f3dasm.datageneration.DataGenerator.experiment_sample.store` method.


class CarStoppingDistance(DataGenerator):
    def __init__(self, mu_z: float, sigma_z: float):
        self.mu_z = mu_z
        self.sigma_z = sigma_z

    def execute(self):
        x = self.experiment_sample.get('x')
        z = norm.rvs(self.mu_z, self.sigma_z, size=1)
        y = z*x + 0.1*x**2
        self.experiment_sample.store(object=y, name='y', to_disk=False)

###############################################################################
# We create an object of the :class:`~CarStoppingDistance` class and pass it to the :meth:`~f3dasm.ExperimentData.evaluate` method:


car_stopping_distance = CarStoppingDistance(mu_z=1.5, sigma_z=0.5)
experiment_data_class.evaluate(
    data_generator=car_stopping_distance, mode='sequential')

print(experiment_data_class)

###############################################################################
#
# There are three methods available of evaluating the experiments:
#
# * :code:`sequential`: regular for-loop over each of the experiments in order
# * :code:`parallel`: utilizing the multiprocessing capabilities (with the `pathos <https://pathos.readthedocs.io/en/latest/pathos.html>`_ multiprocessing library), each experiment is run in a separate core
# * :code:`cluster`: each experiment is run in a seperate node. This is especially useful on a high-performance computation cluster where you have multiple worker nodes and a commonly accessible resource folder. After completion of an experiment, the node will automatically pick the next available open experiment.
# * :code:`cluster_parallel`: Combination of the :code:`cluster` and :code:`parallel` mode. Each node will run multiple samples in parallel.
