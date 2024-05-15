"""
Creating an ExperimentData object from various external sources
===============================================================

The :class:`~f3dasm.ExperimentData` object is the main object used to store implementations of a design-of-experiments, 
keep track of results, perform optimization and extract data for machine learning purposes.

All other processses of :mod:`f3dasm` use this object to manipulate and access data about your experiments.

The :class:`~f3dasm.ExperimentData` object consists of the following attributes:

- :ref:`domain <domain-format>`: The feasible :class:`~f3dasm.design.Domain` of the Experiment. Used for sampling and optimization.
- :ref:`input_data <input-data-format>`: Tabular data containing the input variables of the experiments as column and the experiments as rows.
- :ref:`output_data <output-data-format>`: Tabular data containing the tracked outputs of the experiments.
- :ref:`project_dir <filename-format>`: A user-defined project directory where all files related to your data-driven process will be stored. 
"""

###############################################################################
# The :class:`~f3dasm.ExperimentData` object can be constructed in several ways:
#
# You can construct a :class:`~f3dasm.ExperimentData` object by providing it :ref:`input_data <input-data-format>`,
# :ref:`output_data <output-data-format>`, a :ref:`domain <domain-format>` object and a :ref:`project_dir <filename-format>`.

import numpy as np
import pandas as pd

from f3dasm import ExperimentData
from f3dasm.design import Domain

###############################################################################
# domain
# ^^^^^^
# The domain object is used to define the feasible space of the experiments.

domain = Domain()
domain.add_float('x0', 0., 1.)
domain.add_float('x1', 0., 1.)

###############################################################################
# input_data
# ^^^^^^^^^^
#
# Input data describes the input variables of the experiments.
# The input data is provided in a tabular manner, with the number of rows equal to the number of experiments and the number of columns equal to the number of input variables.
#
# Single parameter values can have any of the basic built-in types: ``int``, ``float``, ``str``, ``bool``. Lists, tuples or array-like structures are not allowed.
#
# We can give the input data as a :class:`~pandas.DataFrame` object with the input variable names as columns and the experiments as rows.

input_data = pd.DataFrame({
    'x0': [0.1, 0.2, 0.3],
    'x1': [0.4, 0.5, 0.6]
})

experimentdata = ExperimentData(domain=domain, input_data=input_data)
print(experimentdata)

###############################################################################
# or a two-dimensional :class:`~numpy.ndarray` object with shape (<number of experiments>, <number of input dimensions>):

input_data = np.array([
    [0.1, 0.4],
    [0.2, 0.5],
    [0.3, 0.6]
])

experimentdata = ExperimentData(domain=domain, input_data=input_data)
print(experimentdata)

###############################################################################
# .. note::
#
#     When providing a :class:`~numpy.ndarray` object, you need to provide a :class:`~f3dasm.design.Domain` object as well.
#     Also, the order of the input variables is inferred from the order of the columns in the :class:`~f3dasm.design.Domain` object.

###############################################################################
# Another option is a path to a ``.csv`` file containing the input data.
# The ``.csv`` file should contain a header row with the names of the input variables
# and the first column should be indices for the experiments.
#
# output_data
# ^^^^^^^^^^^
#
# Output data describes the output variables of the experiments.
# The output data is provided in a tabular manner, with the number of rows equal to the number of experiments and the number of columns equal to the number of output variables.
#
# The same rules apply for the output data as for the input data:

output_data = pd.DataFrame({
    'y': [1.1, 1.2, 1.3],
})

experimentdata = ExperimentData(domain=domain, input_data=input_data,
                                output_data=output_data)

print(experimentdata)

###############################################################################
# .. note::
#
#     When the output to an ExperimentData object is provided, the job will be set to finished,
#     as the output data is considerd the result of the experiment.
