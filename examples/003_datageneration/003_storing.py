"""
Storing data generation output to disk
======================================

After running your simulation, you can store the result back into the :class:`~f3dasm.ExperimentSample` with the :meth:`~f3dasm.ExperimentSample.store` method.
There are two ways of storing your output:

* Singular values can be stored directly to the :attr:`~f3dasm.ExperimentData.output_data`
* Large objects can be stored to disk and a reference path will be stored to the :attr:`~f3dasm.ExperimentData.output_data`.
"""

import numpy as np

from f3dasm import ExperimentData, StoreProtocol
from f3dasm.datageneration import DataGenerator
from f3dasm.design import make_nd_continuous_domain

###############################################################################
# For this example we create a 3 dimensional continuous domain and generate 10 random samples.

domain = make_nd_continuous_domain([[0., 1.], [0., 1.], [0., 1.]])
experiment_data = ExperimentData.from_sampling(
    sampler='random', domain=domain, n_samples=10, seed=42)

###############################################################################
# Single values
# -------------

# Single values or small lists can be stored to the :class:`~f3dasm.ExperimentData` using the ``to_disk=False`` argument, with the name of the parameter as the key.
# This will create a new output parameter if the parameter name is not found in :attr:`~f3dasm.ExperimentData.output_data` of the :class:`~f3dasm.ExperimentData` object:
# This is especially useful if you want to get a quick overview of some loss or design metric of your sample.
#
# We create a custom datagenerator that sums the input features and stores the result back to the :class:`~f3dasm.ExperimentData` object:


class MyDataGenerator_SumInput(DataGenerator):
    def execute(self):
        input_, _ = self.experiment_sample.to_numpy()
        y = float(sum(input_))
        self.experiment_sample.store(object=y, name='y', to_disk=False)

###############################################################################
# We pass the custom data generator to the :meth:`~f3dasm.ExperimentData.evaluate` method and inspect the experimentdata after completion:


my_data_generator_single = MyDataGenerator_SumInput()

experiment_data.evaluate(data_generator=my_data_generator_single)
print(experiment_data)

###############################################################################
#
# All built-in singular types are supported for storing to the :class:`~f3dasm.ExperimentData` this way. Array-like data such as numpy arrays and pandas dataframes are **not** supported and will raise an error.
#
# .. note::
#
#     Outputs stored directly to the :attr:`~f3dasm.ExperimentData.output_data` will be stored within the :class:`~f3dasm.ExperimentData` object.
#     This means that the output will be loaded into memory everytime this object is accessed. For large outputs, it is recommended to store the output to disk.
#
# Large objects and array-like data
# ---------------------------------
#
# In order to store large objects or array-like data, the :meth:`~f3dasm.ExperimentSample.store` method using the ``to_disk=True`` argument, can be used.
# A reference (:code:`Path`) will be saved to the :attr:`~f3dasm.ExperimentData.output_data`.
#
# We create a another custom datagenerator that doubles the input features, but leaves them as an array:

experiment_data = ExperimentData.from_sampling(
    sampler='random', domain=domain, n_samples=10, seed=42)


class MyDataGenerator_DoubleInputs(DataGenerator):
    def execute(self):
        input_, output_ = self.experiment_sample.to_numpy()
        y = input_ * 2
        self.experiment_sample.store(
            object=y, name='output_numpy', to_disk=True)


my_data_generator = MyDataGenerator_DoubleInputs()

experiment_data.evaluate(data_generator=my_data_generator)
print(experiment_data)

###############################################################################
# :mod:`f3dasm` will automatically create a new directory in the project directory for each output parameter and store the object with a generated filename referencing the :attr:`~f3dasm.design.ExperimentSample.job_number` of the design.
#
# .. code-block:: none
#    :caption: Directory Structure
#
#    project_dir/
#    ├── output_numpy/
#    │   ├── 0.npy
#    │   ├── 1.npy
#    │   ├── 2.npy
#    │   └── 3.npy
#    │
#    └── experiment_data/
#        ├── domain.pkl
#        ├── input.csv
#        ├── output.csv
#        └── jobs.pkl
#
#
# In the output data of the :class:`~f3dasm.ExperimentData` object, a reference path (e.g. :code:`/output_numpy/0.npy`) to the stored object will be saved.
#
# Create a custom storage method
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# :mod:`f3dasm` has built-in storing functions for numpy :class:`~numpy.ndarray`, pandas :class:`~pandas.DataFrame` and xarray :class:`~xarray.DataArray` and :class:`~xarray.Dataset` objects.
# For any other type of object, the object will be stored in the `pickle <https://docs.python.org/3/library/pickle.html>`_ format
#
# You can provide your own storing class to the :class:`~f3dasm.ExperimentSample.store` method call:
#
# * a ``store`` method should store an ``self.object`` to disk at the location of ``self.path``
# * a ``load`` method should load the object from disk at the location of ``self.path`` and return it
# * a class variable ``suffix`` should be defined, which is the file extension of the stored object as a string.
# * the class should inherit from the :class:`~f3dasm.StoreProtocol` class
#
# You can take the following class for a :class:`~numpy.ndarray` object as an example:


class NumpyStore(StoreProtocol):
    suffix: int = '.npy'

    def store(self) -> None:
        np.save(file=self.path.with_suffix(self.suffix), arr=self.object)

    def load(self) -> np.ndarray:
        return np.load(file=self.path.with_suffix(self.suffix))

###############################################################################
# After defining the storing function, it can be used as an additional argument in the :meth:`~f3dasm.ExperimentSample.store` method:


class MyDataGenerator_DoubleInputs(DataGenerator):
    def execute(self):
        input_, output_ = self.experiment_sample.to_numpy()
        y = input_ * 2
        self.experiment_sample.store(
            object=y, name='output_numpy',
            to_disk=True, store_method=NumpyStore)
