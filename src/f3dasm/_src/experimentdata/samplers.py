"""Base class for sampling methods"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from itertools import product
from typing import Dict, Literal, Optional, Protocol

# Third-party
import numpy as np
import pandas as pd
from SALib.sample import latin as salib_latin
from SALib.sample import sobol_sequence

# Locals
from ..design.domain import Domain
from ._data import DataTypes

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

SamplerNames = Literal['random', 'latin', 'sobol', 'grid']


class Sampler(Protocol):
    """
    Interface class for samplers
    """
    def __call__(domain: Domain, **kwargs) -> DataTypes:
        ...

#                                                              Factory function
# =============================================================================


def _sampler_factory(sampler: str, domain: Domain) -> Sampler:
    """
    Factory function for samplers

    Parameters
    ----------
    sampler : str
        name of the sampler
    domain : Domain
        domain object

    Returns
    -------
    Sampler
        sampler object
    """
    if sampler.lower() == 'random':
        return randomuniform

    elif sampler.lower() == 'latin':
        return latin

    elif sampler.lower() == 'sobol':
        return sobol

    elif sampler.lower() == 'grid':
        return grid

    else:
        raise KeyError(f"Sampler {sampler} not found!"
                       f"Available built-in samplers are: 'random',"
                       f"'latin' and 'sobol'")


#                                                             Utility functions
# =============================================================================

def _stretch_samples(domain: Domain, samples: np.ndarray) -> np.ndarray:
    """Stretch samples to their boundaries

    Parameters
    ----------
    domain : Domain
        domain object
    samples : np.ndarray
        samples to stretch

    Returns
    -------
    np.ndarray
        stretched samples
    """
    for dim, param in enumerate(domain.space.values()):
        samples[:, dim] = (
            samples[:, dim] * (
                param.upper_bound - param.lower_bound
            ) + param.lower_bound
        )

        # If param.log is True, take the 10** of the samples
        if param.log:
            samples[:, dim] = 10**samples[:, dim]

    return samples


def sample_constant(domain: Domain, n_samples: int):
    """Sample the constant parameters

    Parameters
    ----------
    domain : Domain
        domain object
    n_samples : int
        number of samples

    Returns
    -------
    np.ndarray
        samples
    """
    samples = np.array([param.value for param in domain.space.values()])
    return np.tile(samples, (n_samples, 1))


def sample_np_random_choice(
        domain: Domain, n_samples: int,
        seed: Optional[int] = None, **kwargs):
    """Sample with np random choice

    Parameters
    ----------
    domain : Domain
        domain object
    n_samples : int
        number of samples
    seed : Optional[int], optional
        random seed, by default None

    Returns
    -------
    np.ndarray
        samples
    """
    rng = np.random.default_rng(seed)
    samples = np.empty(shape=(n_samples, len(domain)), dtype=object)
    for dim, param in enumerate(domain.space.values()):
        samples[:, dim] = rng.choice(
            param.categories, size=n_samples)

    return samples


def sample_np_random_choice_range(
        domain: Domain, n_samples: int,
        seed: Optional[int] = None, **kwargs):
    """Samples with np random choice with a range of values

    Parameters
    ----------
    domain : Domain
        domain object
    n_samples : int
        number of samples
    seed : Optional[int], optional
        random seed, by default None

    Returns
    -------
    np.ndarray
        samples
    """
    samples = np.empty(shape=(n_samples, len(domain)), dtype=np.int32)
    rng = np.random.default_rng(seed)
    for dim, param in enumerate(domain.space.values()):
        samples[:, dim] = rng.choice(
            range(param.lower_bound,
                  param.upper_bound + 1, param.step),
            size=n_samples,
        )

    return samples


def sample_np_random_uniform(
        domain: Domain, n_samples: int,
        seed: Optional[int] = None, **kwargs) -> np.ndarray:
    """Sample with numpy random uniform

    Parameters
    ----------
    domain : Domain
        domain object
    n_samples : int
        number of samples
    seed : Optional[int], optional
        random seed, by default None

    Returns
    -------
    np.ndarray
        samples
    """
    rng = np.random.default_rng(seed)
    samples = rng.uniform(low=0.0, high=1.0, size=(n_samples, len(domain)))

    # stretch samples
    samples = _stretch_samples(domain, samples)
    return samples


def sample_latin_hypercube(
        domain: Domain, n_samples: int,
        seed: Optional[int] = None, **kwargs) -> np.ndarray:
    """Sample with Latin Hypercube sampling

    Parameters
    ----------
    domain : Domain
        domain object
    n_samples : int
        number of samples
    seed : Optional[int], optional
        random seed, by default None

    Returns
    -------
    np.ndarray
        samples
    """
    problem = {
        "num_vars": len(domain),
        "names": domain.names,
        "bounds": [[s.lower_bound, s.upper_bound]
                   for s in domain.space.values()],
    }

    samples = salib_latin.sample(problem, N=n_samples, seed=seed)
    return samples


def sample_sobol_sequence(
        domain: Domain, n_samples: int, **kwargs) -> np.ndarray:
    """Sample with Sobol sequence sampling

    Parameters
    ----------
    domain : Domain
        domain object
    n_samples : int
        number of samples

    Returns
    -------
    np.ndarray
        samples
    """
    samples = sobol_sequence.sample(n_samples, len(domain))

    # stretch samples
    samples = _stretch_samples(domain, samples)
    return samples


#                                                             Built-in samplers
# =============================================================================


def randomuniform(
        domain: Domain, n_samples: int, seed: int, **kwargs) -> DataTypes:
    """
    Random uniform sampling

    Parameters
    ----------
    domain : Domain
        domain object
    n_samples : int
        number of samples
    seed : int
        random seed for reproducibility

    Returns
    -------
    DataTypes
        input samples in one of the supported data types for the ExperimentData
        input data.
    """
    _continuous = sample_np_random_uniform(
        domain=domain.continuous, n_samples=n_samples,
        seed=seed)

    _discrete = sample_np_random_choice_range(
        domain=domain.discrete, n_samples=n_samples,
        seed=seed)

    _categorical = sample_np_random_choice(
        domain=domain.categorical, n_samples=n_samples,
        seed=seed)

    _constant = sample_constant(domain.constant, n_samples)

    df = pd.concat(
        [pd.DataFrame(_continuous, columns=domain.continuous.names),
         pd.DataFrame(_discrete, columns=domain.discrete.names),
         pd.DataFrame(
            _categorical, columns=domain.categorical.names),
         pd.DataFrame(_constant, columns=domain.constant.names)], axis=1
    )[domain.names]

    return df


def grid(
    domain: Domain, stepsize_continuous_parameters:
        Optional[Dict[str, float] | float] = None, **kwargs) -> DataTypes:
    """Receive samples of the search space

    Parameters
    ----------
    n_samples : int
        number of samples
    stepsize_continuous_parameters : Dict[str, float] | float, optional
        stepsize for the continuous parameters, by default None.
        If a float is given, all continuous parameters are sampled with
        the same stepsize. If a dictionary is given, the stepsize for each
        continuous parameter can be specified.

    Returns
    -------
    DataTypes
        input samples in one of the supported data types for the ExperimentData
        input data.

    Raises
    ------
    ValueError
        If the stepsize_continuous_parameters is given as a dictionary
        and not specified for all continuous parameters.
    """
    continuous = domain.continuous

    if not continuous.space:
        discrete_space = continuous.space

    elif isinstance(stepsize_continuous_parameters, (float, int)):
        discrete_space = {name: param.to_discrete(
            step=stepsize_continuous_parameters)
            for name, param in continuous.space.items()}

    elif isinstance(stepsize_continuous_parameters, dict):
        discrete_space = {key: continuous.space[key].to_discrete(
            step=value) for key,
            value in stepsize_continuous_parameters.items()}

        if len(discrete_space) != len(domain.continuous):
            raise ValueError(
                "If you specify the stepsize for continuous parameters, \
                the stepsize_continuous_parameters should \
                contain all continuous parameters")

    continuous_to_discrete = Domain(discrete_space)

    _iterdict = {}

    for k, v in domain.categorical.space.items():
        _iterdict[k] = v.categories

    for k, v, in domain.discrete.space.items():
        _iterdict[k] = range(v.lower_bound, v.upper_bound+1, v.step)

    for k, v, in continuous_to_discrete.space.items():
        _iterdict[k] = np.arange(
            start=v.lower_bound, stop=v.upper_bound, step=v.step)

    df = pd.DataFrame(list(product(*_iterdict.values())),
                      columns=_iterdict, dtype=object)[domain.names]

    return df


def sobol(domain: Domain, n_samples: int, seed: int, **kwargs) -> DataTypes:
    """
    Sobol sequence sampling

    Parameters
    ----------
    domain : Domain
        domain object
    n_samples : int
        number of samples
    seed : int
        random seed for reproducibility

    Returns
    -------
    DataTypes
        input samples in one of the supported data types for the ExperimentData
        input data.
    """
    _continuous = sample_sobol_sequence(
        domain=domain.continuous, n_samples=n_samples)

    _discrete = sample_np_random_choice_range(
        domain=domain.discrete, n_samples=n_samples, seed=seed)

    _categorical = sample_np_random_choice(
        domain=domain.categorical, n_samples=n_samples, seed=seed)

    _constant = sample_constant(domain=domain.constant, n_samples=n_samples)

    df = pd.concat(
        [pd.DataFrame(_continuous, columns=domain.continuous.names),
         pd.DataFrame(_discrete, columns=domain.discrete.names),
         pd.DataFrame(
            _categorical, columns=domain.categorical.names),
         pd.DataFrame(_constant, columns=domain.constant.names)], axis=1
    )[domain.names]

    return df


def latin(domain: Domain, n_samples: int, seed: int, **kwargs) -> DataTypes:
    """
    Latin Hypercube sampling

    Parameters
    ----------
    domain : Domain
        domain object
    n_samples : int
        number of samples
    seed : int
        random seed for reproducibility

    Returns
    -------
    DataTypes
        input samples in one of the supported data types for the ExperimentData
        input data.
    """
    _continuous = sample_latin_hypercube(
        domain=domain.continuous, n_samples=n_samples, seed=seed)

    _discrete = sample_np_random_choice_range(
        domain=domain.discrete, n_samples=n_samples, seed=seed)

    _categorical = sample_np_random_choice(
        domain=domain.categorical, n_samples=n_samples, seed=seed)

    _constant = sample_constant(domain=domain.constant, n_samples=n_samples)

    df = pd.concat(
        [pd.DataFrame(_continuous, columns=domain.continuous.names),
         pd.DataFrame(_discrete, columns=domain.discrete.names),
         pd.DataFrame(
            _categorical, columns=domain.categorical.names),
         pd.DataFrame(_constant, columns=domain.constant.names)], axis=1
    )[domain.names]

    return df
