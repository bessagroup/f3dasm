"""
Implementing a grid search sampler from scratch
===============================================

In this example, we will implement a `grid search sampler <https://en.wikipedia.org/wiki/Hyperparameter_optimization>`_ from scratch.
The grid search sampler is a simple sampler that evaluates all possible combinations of the parameters in the domain. This is useful for small domains, but it can become computationally expensive for larger domains.
We will show how to create this sampler and use it in a :mod:`f3dasm` data-driven experiment.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, Optional

import numpy as np
import pandas as pd

from f3dasm import ExperimentData
from f3dasm.design import Domain

###############################################################################
# When integrating your sampling strategy into the data-driven process, you have to create a function that will take the domain as argument:
# Several other optional arguments can be passed to the function as well, such as:
#
# * :code:`n_samples`: The number of samples you wish to generate. It's not always necessary to define this upfront, as some sampling methods might inherently determine the number of samples based on certain criteria or properties of the domain.
# * :code:`seed`: A seed for the random number generator to replicate the sampling process. This enables you to control the randomness of the sampling process [1]_. By setting a seed, you ensure reproducibility, meaning that if you run the sampling function with the same seed multiple times, you'll get the same set of samples.
#
# .. [1] If no seed is provided, the function should use a random seed.
#
# Additionally, the function can accept any other keyword arguments that you might need to pass to the sampling function.
# This also means that you shoud handle arbitrary keyword arguments in your function with the `**kwargs` syntax.
# The function should return the samples (``input_data``) in one of the following formats:
#
# * A :class:`~pandas.DataFrame` object
# * A :class:`~numpy.ndarray` object
#
# For our implementation of the grid-search sampler, we need to handle the case of continous parameters.
# We require the user to pass down a dictionary with the discretization stepsize for each continuous parameter.


def grid(
    domain: Domain, stepsize_continuous_parameters:
        Optional[Dict[str, float] | float] = None, **kwargs) -> pd.DataFrame:

    # Extract the continous part of the domain
    continuous = domain.continuous

    # If therei s no continuos space, we can return an empty dictionary
    if not continuous.input_space:
        discrete_space = {}

    else:
        discrete_space = {key: continuous.input_space[key].to_discrete(
            step=value) for key,
            value in stepsize_continuous_parameters.items()}

    continuous_to_discrete = Domain(discrete_space)

    _iterdict = {}

    # For all the categorical parameters, we will iterate over the categories
    for k, v in domain.categorical.input_space.items():
        _iterdict[k] = v.categories

    # For all the discrete parameters, we will iterate over the range of values
    for k, v, in domain.discrete.input_space.items():
        _iterdict[k] = range(v.lower_bound, v.upper_bound+1, v.step)

    # For all the continuous parameters, we will iterate over the range of values
    # based on the stepsize provided
    for k, v, in continuous_to_discrete.input_space.items():
        _iterdict[k] = np.arange(
            start=v.lower_bound, stop=v.upper_bound, step=v.step)

    # We will create a dataframe with all the possible combinations using
    # the itertools.product function
    df = pd.DataFrame(list(product(*_iterdict.values())),
                      columns=_iterdict, dtype=object)[domain.names]

    # return the samples
    return df

###############################################################################
# To test our implementation, we will create a domain with a mix of continuous, discrete, and categorical parameters.


domain = Domain()
domain.add_float("param_1", -1.0, 1.0)
domain.add_int("param_2", 1, 5)
domain.add_category("param_3", ["red", "blue", "green", "yellow", "purple"])

###############################################################################
# We will now sample the domain using the grid sampler we implemented.
# We can create an empty ExperimentData object and call the :meth:`~f3dasm.data.ExperimentData.sample` method to add the samples to the object:

experiment_data = ExperimentData(domain=domain)
experiment_data.sample(
    sampler=grid, stepsize_continuous_parameters={"param_1": 0.2})

###############################################################################
# We can print the samples to see the results:

print(experiment_data)
