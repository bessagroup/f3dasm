"""Base class for sampling methods"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from itertools import product
from math import ceil, prod
from typing import Literal, Optional

# Third-party
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from SALib.sample import latin as salib_latin
from SALib.sample import sobol as salib_sobol
from SALib.sample import sobol_sequence

# Locals
from .design.domain import Domain, Parameter
from .design.parameter import (
    ArrayParameter,
    CategoricalParameter,
    ConstantParameter,
    ContinuousParameter,
    DiscreteParameter,
)
from .experimentdata import Block, ExperimentData
from .experimentsample import ExperimentSample

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


class Sampler(Block):
    sample_mapping: dict[Parameter, callable]

    def __init__(self, seed: Optional[int], **parameters):
        self.seed = seed
        self.parameters = parameters

    def call(
        self, data: ExperimentData, n_samples: int, **kwargs
    ) -> ExperimentData:
        d = ExperimentData(
            project_dir=data._project_dir, domain=data.domain._copy()
        )

        for param_type, sampler_func in self.sample_mapping.items():
            filtered_domain = d.domain._filter(param_type)
            if filtered_domain.input_space:
                experiment_samples: dict[int, ExperimentSample] = sampler_func(
                    input_space=filtered_domain.input_space,
                    n_samples=n_samples,
                    seed=self.seed,
                    **self.parameters,
                )

                samples = ExperimentData.from_data(
                    data=experiment_samples,
                    domain=filtered_domain,
                    project_dir=data._project_dir,
                )

                d = d.join(samples)
        return d


#                                                               Random Sampling
# =============================================================================


def random_sample_continuous_parameters(
    input_space: dict[str, ContinuousParameter],
    n_samples: int,
    seed: Optional[int] = None,
    **kwargs,
) -> dict[int, ExperimentSample]:
    """
    Sample continuous parameters randomly.

    Parameters
    ----------
    input_space : dict[str, ContinuousParameter]
        A dictionary mapping parameter names to ContinuousParameter objects.
    n_samples : int
        The number of samples to generate.
    seed : Optional[int], optional
        The random seed, by default None

    Returns
    -------
    dict[int, ExperimentSample]
        A dictionary mapping sample indices to ExperimentSample objects
        containing the sampled values.
    """
    rng = np.random.default_rng(seed)
    samples = {}
    for i in range(n_samples):
        sample_values = {
            name: rng.uniform(low=param.lower_bound, high=param.upper_bound)
            for name, param in input_space.items()
        }
        samples[i] = ExperimentSample(_input_data=sample_values)

    return samples


def random_sample_discrete_parameters(
    input_space: dict[str, DiscreteParameter],
    n_samples: int,
    seed: Optional[int] = None,
    **kwargs,
) -> dict[int, ExperimentSample]:
    """
    Sample discrete parameters randomly.

    Parameters
    ----------
    input_space : dict[str, DiscreteParameter]
        A dictionary mapping parameter names to DiscreteParameter objects.
    n_samples : int
        The number of samples to generate.
    seed : Optional[int], optional
        The random seed, by default None

    Returns
    -------
    dict[int, ExperimentSample]
        A dictionary mapping sample indices to ExperimentSample objects
        containing the sampled values.
    """
    rng = np.random.default_rng(seed)
    samples = {}
    for i in range(n_samples):
        sample_values = {
            name: rng.choice(
                range(param.lower_bound, param.upper_bound + 1, param.step)
            )
            for name, param in input_space.items()
        }
        samples[i] = ExperimentSample(_input_data=sample_values)

    return samples


def random_sample_categorical_parameters(
    input_space: dict[str, CategoricalParameter],
    n_samples: int,
    seed: Optional[int] = None,
    **kwargs,
) -> dict[int, ExperimentSample]:
    """
    Sample categorical parameters randomly.

    Parameters
    ----------
    input_space : dict[str, CategoricalParameter]
        A dictionary mapping parameter names to CategoricalParameter objects.
    n_samples : int
        The number of samples to generate.
    seed : Optional[int], optional
        The random seed, by default None

    Returns
    -------
    dict[int, ExperimentSample]
        A dictionary mapping sample indices to ExperimentSample objects
        containing the sampled values.
    """
    rng = np.random.default_rng(seed)
    samples = {}
    for i in range(n_samples):
        sample_values = {
            name: rng.choice(param.categories)
            for name, param in input_space.items()
        }
        samples[i] = ExperimentSample(_input_data=sample_values)

    return samples


def random_sample_constant_parameters(
    input_space: dict[str, ConstantParameter],
    n_samples: int,
    **kwargs,
) -> dict[int, ExperimentSample]:
    """
    Sample constant parameters.

    Parameters
    ----------
    input_space : dict[str, ConstantParameter]
        A dictionary mapping parameter names to ConstantParameter objects.
    n_samples : int
        The number of samples to generate.
    seed : Optional[int], optional
        The random seed, by default None

    Returns
    -------
    dict[int, ExperimentSample]
        A dictionary mapping sample indices to ExperimentSample objects
        containing the sampled values.
    """
    samples = {}
    for i in range(n_samples):
        sample_values = {
            name: param.value for name, param in input_space.items()
        }
        samples[i] = ExperimentSample(_input_data=sample_values)

    return samples


def random_sample_array_parameters(
    input_space: dict[str, ArrayParameter],
    n_samples: int,
    seed: Optional[int] = None,
    **kwargs,
) -> dict[int, ExperimentSample]:
    """
    Sample array parameters randomly.

    Parameters
    ----------
    input_space : dict[str, ArrayParameter]
        A dictionary mapping parameter names to ArrayParameter objects.
    n_samples : int
        The number of samples to generate.
    seed : Optional[int], optional
        The random seed, by default None

    Returns
    -------
    dict[int, ExperimentSample]
        A dictionary mapping sample indices to ExperimentSample objects
        containing the sampled values.
    """
    rng = np.random.default_rng(seed)
    samples = {}
    for i in range(n_samples):
        sample_values = {
            name: rng.uniform(
                low=param.lower_bound, high=param.upper_bound, size=param.shape
            )
            for name, param in input_space.items()
        }
        samples[i] = ExperimentSample(_input_data=sample_values)

    return samples


random_sample_mapping: dict[Parameter, callable] = {
    ContinuousParameter: random_sample_continuous_parameters,
    DiscreteParameter: random_sample_discrete_parameters,
    CategoricalParameter: random_sample_categorical_parameters,
    ConstantParameter: random_sample_constant_parameters,
    ArrayParameter: random_sample_array_parameters,
}


class RandomUniform(Sampler):
    sample_mapping: dict[Parameter, callable] = random_sample_mapping


def random(seed: Optional[int] = None, **kwargs) -> Block:
    """
    Create a RandomUniform sampler.

    Parameters
    ----------
    seed : Optional[int], optional
        The random seed, by default None
    **kwargs : dict
        Additional parameters for the sampler.

    Returns
    -------
    Block
        An Block instance of a random uniform sampler.
    """
    return RandomUniform(seed=seed, **kwargs)


# =============================================================================


def latin_sample_continuous_parameters(
    input_space: dict[str, ContinuousParameter],
    n_samples: int,
    seed: Optional[int] = None,
    **kwargs,
) -> dict[int, ExperimentSample]:
    """
    Sample continuous parameters with Latin Hypercube sampling.

    Parameters
    ----------
    input_space : dict[str, ContinuousParameter]
        A dictionary mapping parameter names to ContinuousParameter objects.
    n_samples : int
        The number of samples to generate.
    seed : Optional[int], optional
        The random seed, by default None

    Returns
    -------
    dict[int, ExperimentSample]
        A dictionary mapping sample indices to ExperimentSample objects
        containing the sampled values.
    """
    # This is just a wrapper around the sample_latin_hypercube function that
    # converts the output to the desired format.
    problem = {
        "num_vars": len(input_space),
        "names": list(input_space.keys()),
        "bounds": [
            [param.lower_bound, param.upper_bound]
            for param in input_space.values()
        ],
    }

    samples_array = salib_latin.sample(problem=problem, N=n_samples, seed=seed)

    samples = {}
    for i in range(n_samples):
        sample_values = {
            name: samples_array[i, idx]
            for idx, name in enumerate(input_space.keys())
        }
        samples[i] = ExperimentSample(_input_data=sample_values)

    return samples


def latin_sample_array_parameters(
    input_space: dict[str, ArrayParameter],
    n_samples: int,
    seed: Optional[int] = None,
    **kwargs,
) -> dict[int, ExperimentSample]:
    """
    Sample array parameters with Latin Hypercube sampling.

    Parameters
    ----------
    input_space : dict[str, ArrayParameter]
        A dictionary mapping parameter names to ArrayParameter objects.
    n_samples : int
        The number of samples to generate.
    seed : Optional[int], optional
        The random seed, by default None

    Returns
    -------
    dict[int, ExperimentSample]
        A dictionary mapping sample indices to ExperimentSample objects
        containing the sampled values.
    """
    samples = []
    for name, param in input_space.items():
        lb = np.ravel(param.lower_bound)
        ub = np.ravel(param.upper_bound)
        problem = {
            "num_vars": prod(param.shape),
            "names": [f"{name}{i}" for i in range(prod(param.shape))],
            "bounds": list(zip(lb, ub, strict=False)),  # per-dimension bounds
        }

        s = salib_latin.sample(problem=problem, N=n_samples, seed=seed)
        s = s.reshape((n_samples,) + param.shape)
        samples.append(s)

    samples_dict = {}
    for i in range(n_samples):
        sample_values = {
            name: samples[idx][i]
            for idx, name in enumerate(input_space.keys())
        }
        samples_dict[i] = ExperimentSample(_input_data=sample_values)

    return samples_dict


latin_sample_mapping: dict[Parameter, callable] = {
    ContinuousParameter: latin_sample_continuous_parameters,
    DiscreteParameter: random_sample_discrete_parameters,
    CategoricalParameter: random_sample_categorical_parameters,
    ConstantParameter: random_sample_constant_parameters,
    ArrayParameter: latin_sample_array_parameters,
}


class Latin(Sampler):
    sample_mapping: dict[Parameter, callable] = latin_sample_mapping


def latin(seed: Optional[int] = None, **kwargs) -> Block:
    """
    Create a Latin Hypercube sampler.

    Parameters
    ----------
    seed : Optional[int], optional
        The random seed, by default None
    **kwargs : dict
        Additional parameters for the sampler.

    Returns
    -------
    Block
        An Block instance of a latin hypercube sampler.
    """
    return Latin(seed=seed, **kwargs)


#                                                                         Sobol
# =============================================================================


def sobol_sample_continuous_parameters(
    input_space: dict[str, ContinuousParameter],
    n_samples: int,
    seed: Optional[int] = None,
    **kwargs,
) -> dict[int, ExperimentSample]:
    """
    Sample continuous parameters with Sobol sequence sampling.

    Parameters
    ----------
    input_space : dict[str, ContinuousParameter]
        A dictionary mapping parameter names to ContinuousParameter objects.
    n_samples : int
        The number of samples to generate.
    seed : Optional[int], optional
        The random seed, by default None

    Returns
    -------
    dict[int, ExperimentSample]
        A dictionary mapping sample indices to ExperimentSample objects
        containing the sampled values.
    """
    sobol_seq = sobol_sequence.sample(N=n_samples, D=len(input_space))

    lower = np.array([input_space[name].lower_bound for name in input_space])
    upper = np.array([input_space[name].upper_bound for name in input_space])
    samples_array = lower + sobol_seq * (upper - lower)

    samples = {}
    for i in range(n_samples):
        sample_values = {
            name: samples_array[i, idx]
            for idx, name in enumerate(input_space.keys())
        }
        samples[i] = ExperimentSample(_input_data=sample_values)

    return samples


def sobol_sample_array_parameters(
    input_space: dict[str, ArrayParameter],
    n_samples: int,
    seed: Optional[int] = None,
    **kwargs,
) -> dict[int, ExperimentSample]:
    """
    Sample array parameters with Sobol sequence sampling.

    Parameters
    ----------
    input_space : dict[str, ArrayParameter]
        A dictionary mapping parameter names to ArrayParameter objects.
    n_samples : int
        The number of samples to generate.
    seed : Optional[int], optional
        The random seed, by default None

    Returns
    -------
    dict[int, ExperimentSample]
        A dictionary mapping sample indices to ExperimentSample objects
        containing the sampled values.
    """
    samples = []
    for name, param in input_space.items():
        lb = np.ravel(param.lower_bound)
        ub = np.ravel(param.upper_bound)
        problem = {
            "num_vars": prod(param.shape),
            "names": [f"{name}{i}" for i in range(prod(param.shape))],
            "bounds": list(zip(lb, ub, strict=False)),
        }

        N = next_power_of_two(ceil(n_samples / (prod(param.shape) + 2)))

        s = salib_sobol.sample(
            problem=problem,
            N=N,
            seed=seed,
            calc_second_order=False,
        )

        s = s[:n_samples].reshape((n_samples,) + param.shape)
        samples.append(s)

    samples_dict = {}
    for i in range(n_samples):
        sample_values = {
            name: samples[idx][i]
            for idx, name in enumerate(input_space.keys())
        }
        samples_dict[i] = ExperimentSample(_input_data=sample_values)

    return samples_dict


sobol_sample_mapping: dict[Parameter, callable] = {
    ContinuousParameter: sobol_sample_continuous_parameters,
    DiscreteParameter: random_sample_discrete_parameters,
    CategoricalParameter: random_sample_categorical_parameters,
    ConstantParameter: random_sample_constant_parameters,
    ArrayParameter: sobol_sample_array_parameters,
}


class Sobol(Sampler):
    sample_mapping: dict[Parameter, callable] = sobol_sample_mapping


def sobol(seed: Optional[int] = None, **kwargs) -> Block:
    """
    Create a Sobol sampler.

    Parameters
    ----------
    seed : Optional[int], optional
        The random seed, by default None
    **kwargs : dict
        Additional parameters for the sampler.

    Returns
    -------
    Block
        A Block instance of a sobol sequence sampler.
    """
    return Sobol(seed=seed, **kwargs)


#                                                             Utility functions
# =============================================================================


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two greater than or equal to x.

    Parameters
    ----------
    x : int
        A positive integer.

    Returns
    -------
    int
        The smallest power of two that is >= x.
    """
    return 1 if x <= 1 else 2 ** (x - 1).bit_length()


# =============================================================================


class Grid(Block):
    def __init__(self, **parameters):
        """
        Initialize the Grid sampler.

        Parameters
        ----------
        **parameters : dict
            Additional parameters for the sampler.
        """
        self.parameters = parameters

    def call(
        self,
        data: ExperimentData,
        stepsize_continuous_parameters: Optional[
            dict[str, float] | float
        ] = None,
        **kwargs,
    ) -> ExperimentData:
        """
        Sample data using the Grid method.

        Parameters
        ----------
        data : ExperimentData
            The experiment data object providing the domain and project dir.
        stepsize_continuous_parameters : dict[str, float] or float, optional
            Step size for continuous parameters. If a single float, the same
            step size is used for all continuous parameters. If a dict, maps
            parameter names to individual step sizes.
        **kwargs : dict
            Additional parameters for sampling.

        Returns
        -------
        ExperimentData
            A new ExperimentData object containing the sampled input data.
        """
        continuous = data.domain.continuous

        if not continuous.input_space:
            discrete_space = continuous.input_space

        elif isinstance(stepsize_continuous_parameters, float | int):
            discrete_space = {
                name: param.to_discrete(step=stepsize_continuous_parameters)
                for name, param in continuous.input_space.items()
            }

        elif isinstance(stepsize_continuous_parameters, dict):
            discrete_space = {
                key: continuous.input_space[key].to_discrete(step=value)
                for key, value in stepsize_continuous_parameters.items()
            }

            if len(discrete_space) != len(data.domain.continuous):
                raise ValueError(
                    "If you specify the stepsize for continuous parameters, \
                    the stepsize_continuous_parameters should \
                    contain all continuous parameters"
                )

        continuous_to_discrete = Domain(discrete_space)

        _iterdict = {}

        for k, v in data.domain.categorical.input_space.items():
            _iterdict[k] = v.categories

        for (
            k,
            v,
        ) in data.domain.discrete.input_space.items():
            _iterdict[k] = range(v.lower_bound, v.upper_bound + 1, v.step)

        for (
            k,
            v,
        ) in continuous_to_discrete.input_space.items():
            _iterdict[k] = np.arange(
                start=v.lower_bound, stop=v.upper_bound, step=v.step
            )

        for (
            k,
            v,
        ) in data.domain.constant.input_space.items():
            _iterdict[k] = [v.value]

        df = pd.DataFrame(
            list(product(*_iterdict.values())), columns=_iterdict, dtype=object
        )[data.domain.input_names]

        return ExperimentData(
            domain=data.domain._copy(),
            input_data=df,
            project_dir=data._project_dir,
        )


def grid(**kwargs) -> Block:
    """
    Create a Grid sampler.

    Parameters
    ----------
    **kwargs : dict
        Additional parameters for the sampler.

    Returns
    -------
    Block
        A Block instance of a grid sampler.
    """
    return Grid(**kwargs)


# =============================================================================


_SAMPLERS = [random, latin, sobol, grid]

SAMPLER_MAPPING: dict[str, Block] = {
    sampler.__name__.lower(): sampler for sampler in _SAMPLERS
}

#                                                              Factory function
# =============================================================================


def create_sampler(sampler: str | DictConfig, **parameters) -> Block:
    """
    Create a sampler block from one of the built-in samplers.

    Parameters
    ----------
    sampler : str | Block | DictConfig
        name of the built-in sampler. This can be a string with the name of the
        sampler, a Block object (this will just by-pass the function), or a
        DictConfig object (the sampler will be instantiated with
        hydra.instantiate).
    **parameters
        Additional keyword arguments passed when initializing the sampler

    Returns
    -------
    Block
        Block object of the sampler

    Raises
    ------
    KeyError
        If the built-in sampler name is not recognized.
    TypeError
        If the given type is not recognized.
    """
    if isinstance(sampler, str):
        filtered_name = (
            sampler.lower().replace(" ", "").replace("-", "").replace("_", "")
        )

        if filtered_name in SAMPLER_MAPPING:
            return SAMPLER_MAPPING[filtered_name](**parameters)

        else:
            raise KeyError(f"Unknown built-in sampler name: {sampler}")

    elif isinstance(sampler, DictConfig):
        return instantiate(sampler)

    else:
        raise TypeError(f"Unknown sampler type given: {type(sampler)}")


SamplerNames = Literal["random", "latin", "sobol", "grid"]
