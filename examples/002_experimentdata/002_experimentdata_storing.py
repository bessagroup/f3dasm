"""
Storing experiment data to disk
===============================

In this example, we will show how to store the experiment data to disk using the :meth:`~f3dasm.ExperimentData.store` method and
how to load the stored data using the :meth:`~f3dasm.ExperimentData.from_file` method.
"""

###############################################################################
# project directory
# ^^^^^^^^^^^^^^^^^
#
# The ``project_dir`` argument is used to store the ExperimentData to disk
# You can provide a string or a path to a directory. This can either be a relative or absolute path.
# If the directory does not exist, it will be created.

from f3dasm import ExperimentData

data = ExperimentData()
data.set_project_dir("./example_project_dir")

print(data.project_dir)

###############################################################################
# Storing the data
# ^^^^^^^^^^^^^^^^^
#
# The :meth:`~f3dasm.ExperimentData.store` method is used to store the experiment data to disk.

data.store()

###############################################################################
# The data is stored in several files in an 'experiment_data' subfolder in the provided project directory:
#
# .. code-block:: none
#    :caption: Directory Structure
#
#    my_project/
#    ├── my_script.py
#    └── experiment_data
#          ├── domain.pkl
#          ├── input_data.csv
#          ├── output_data.csv
#          └── jobs.pkl
#
# In order to load the data, you can use the :meth:`~f3dasm.ExperimentData.from_file` method.

data_loaded = ExperimentData.from_file(project_dir="./example_project_dir")

print(data_loaded)
