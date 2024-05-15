"""
Combine hydra configurations with f3dasm
========================================

.. _hydra: https://hydra.cc/

`hydra <https://hydra.cc/>`_ is an open-source configuration management framework that is widely used in machine learning and other software development domains.
It is designed to help developers manage and organize complex configuration settings for their projects,
making it easier to experiment with different configurations, manage multiple environments, and maintain reproducibility in their work.

`hydra <https://hydra.cc/>`_ can be seamlessly integrated with the worfklows in :mod:`f3dasm` to manage the configuration settings for the project.
"""

from hydra import compose, initialize

from f3dasm import ExperimentData
from f3dasm.design import Domain

###############################################################################
# Domain from a `hydra <https://hydra.cc/>`_ configuration file
# -------------------------------------------------------------
#
# If you are using `hydra <https://hydra.cc/>`_ to manage your configuration files, you can create a domain from a configuration file.
# Your config needs to have a key (e.g. :code:`domain`) that has a dictionary with the parameter names (e.g. :code:`param_1`) as keys
# and a dictionary with the parameter type (:code:`type`) and the corresponding arguments as values:
#
# .. code-block:: yaml
#    :caption: config.yaml
#
#     domain:
#         param_1:
#             type: float
#             low: -1.0
#             high: 1.0
#         param_2:
#             type: int
#             low: 1
#             high: 10
#         param_3:
#             type: category
#             categories: ['red', 'blue', 'green', 'yellow', 'purple']
#         param_4:
#             type: constant
#             value: some_value
#
# In order to run the following code snippet, you need to have a configuration file named :code:`config.yaml` in the current working directory.


with initialize(version_base=None, config_path="."):
    config = compose(config_name="config")

domain = Domain.from_yaml(config.domain)
print(domain)

###############################################################################
# ExperimentData from a `hydra <https://hydra.cc/>`_ configuration file
# ---------------------------------------------------------------------
#
# If you are using `hydra <https://hydra.cc/>`_ for configuring your experiments, you can use it to construct
# an :class:`~f3dasm.ExperimentData` object from the information in the :code:`config.yaml` file with the :meth:`~f3dasm.ExperimentData.from_yaml` method.
#
# ExperimentData from file with hydra
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# You can create an experimentdata :class:`~f3dasm.ExperimentData` object in the same way as the :meth:`~f3dasm.design.Domain.from_file` method, but with the :code:`from_file` key in the :code:`config.yaml` file:
#
# .. code-block:: yaml
#     :caption: config_from_file.yaml
#
#     domain:
#         x0:
#             type: float
#             lower_bound: 0.
#             upper_bound: 1.
#         x1:
#             type: float
#             lower_bound: 0.
#             upper_bound: 1.
#
#     experimentdata:
#         from_file: ./example_project_dir
#
# .. note::
#
#     The :class:`~f3dasm.design.Domain` object will be constructed using the :code:`domain` key in the :code:`config.yaml` file. Make sure you have the :code:`domain` key in your :code:`config.yaml`!
#
#
# Inside your python script, you can then create the :class:`~f3dasm.ExperimentData` object with the :meth:`~f3dasm.ExperimentData.from_yaml` method:

with initialize(version_base=None, config_path="."):
    config_from_file = compose(config_name="config_from_file")

data_from_file = ExperimentData.from_yaml(config_from_file.experimentdata)
print(data_from_file)

###############################################################################
# ExperimentData from sampling with hydra
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To create the :class:`~f3dasm.ExperimentData` object with the :meth:`~f3dasm.ExperimentData.from_sampling` method,
# you can use the following configuration:
#
# .. code-block:: yaml
#    :caption: config_from_sampling.yaml
#
#     domain:
#         x0:
#             type: float
#             lower_bound: 0.
#             upper_bound: 1.
#         x1:
#             type: float
#             lower_bound: 0.
#             upper_bound: 1.
#
#     experimentdata:
#         from_sampling:
#             domain: ${domain}
#             sampler: random
#             seed: 1
#             n_samples: 10
#
# In order to run the following code snippet, you need to have a configuration file named :code:`config_from_sampling.yaml` in the current working directory.

with initialize(version_base=None, config_path="."):
    config_sampling = compose(config_name="config_from_sampling")

data_from_sampling = ExperimentData.from_yaml(config_sampling.experimentdata)
print(data_from_sampling)

###############################################################################
# Combining both approaches
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# You can also combine both approaches to create the :class:`~f3dasm.ExperimentData` object by
# continuing an existing experiment with new samples. This can be done by providing both keys:
#
# .. code-block:: yaml
#    :caption: config_combining.yaml
#
#     domain:
#         x0:
#             type: float
#             lower_bound: 0.
#             upper_bound: 1.
#         x1:
#             type: float
#             lower_bound: 0.
#             upper_bound: 1.
#
#     experimentdata:
#         from_file: ./example_project_dir
#         from_sampling:
#             domain: ${domain}
#             sampler: random
#             seed: 1
#             n_samples: 10
#
# In order to run the following code snippet, you need to have a configuration file named :code:`config_combining.yaml` in the current working directory.

with initialize(version_base=None, config_path="."):
    config_sampling = compose(config_name="config_combining")

data_combining = ExperimentData.from_yaml(config_sampling.experimentdata)
print(data_combining)
