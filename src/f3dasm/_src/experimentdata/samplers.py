"""Base class for sampling methods"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from abc import abstractmethod
from itertools import product
from typing import Dict, Literal, Optional

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


class Sampler:
    def init(self, domain: Domain):
        self.domain = domain

    @abstractmethod
    def sample(self, **kwargs) -> DataTypes:
        ...

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
    for dim, param in enumerate(domain.input_space.values()):
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
    samples = np.array([param.value for param in domain.input_space.values()])
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
    for dim, param in enumerate(domain.input_space.values()):
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
    for dim, param in enumerate(domain.input_space.values()):
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
                   for s in domain.input_space.values()],
    }

    samples = salib_latin.sample(problem=problem, N=n_samples, seed=seed)
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
    samples = sobol_sequence.sample(N=n_samples, D=len(domain))

    # stretch samples
    samples = _stretch_samples(domain, samples)
    return samples


#                                                             Built-in samplers
# =============================================================================

class RandomUniform(Sampler):
    def __init__(self, seed: Optional[int], **parameters):
        self.seed = seed
        self.parameters = parameters

    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        _continuous = sample_np_random_uniform(
            domain=self.domain.continuous, n_samples=n_samples,
            seed=self.seed)

        _discrete = sample_np_random_choice_range(
            domain=self.domain.discrete, n_samples=n_samples,
            seed=self.seed)

        _categorical = sample_np_random_choice(
            domain=self.domain.categorical, n_samples=n_samples,
            seed=self.seed)

        _constant = sample_constant(self.domain.constant, n_samples)

        df = pd.concat(
            [pd.DataFrame(_continuous, columns=self.domain.continuous.names),
             pd.DataFrame(_discrete, columns=self.domain.discrete.names),
             pd.DataFrame(
                _categorical, columns=self.domain.categorical.names),
             pd.DataFrame(_constant, columns=self.domain.constant.names)],
            axis=1
        )[self.domain.names]

        return df


def random(seed: Optional[int] = None, **kwargs) -> Sampler:
    return RandomUniform(seed=seed, **kwargs)


# =============================================================================

class Grid(Sampler):
    def __init__(self, **parameters):
        self.parameters = parameters

    def sample(self,
               stepsize_continuous_parameters:
               Optional[Dict[str, float] | float],
               **kwargs) -> pd.DataFrame:
        continuous = self.domain.continuous

        if not continuous.input_space:
            discrete_space = continuous.input_space

        elif isinstance(stepsize_continuous_parameters, (float, int)):
            discrete_space = {name: param.to_discrete(
                step=stepsize_continuous_parameters)
                for name, param in continuous.input_space.items()}

        elif isinstance(stepsize_continuous_parameters, dict):
            discrete_space = {key: continuous.input_space[key].to_discrete(
                step=value) for key,
                value in stepsize_continuous_parameters.items()}

            if len(discrete_space) != len(self.domain.continuous):
                raise ValueError(
                    "If you specify the stepsize for continuous parameters, \
                    the stepsize_continuous_parameters should \
                    contain all continuous parameters")

        continuous_to_discrete = Domain(discrete_space)

        _iterdict = {}

        for k, v in self.domain.categorical.input_space.items():
            _iterdict[k] = v.categories

        for k, v, in self.domain.discrete.input_space.items():
            _iterdict[k] = range(v.lower_bound, v.upper_bound+1, v.step)

        for k, v, in continuous_to_discrete.input_space.items():
            _iterdict[k] = np.arange(
                start=v.lower_bound, stop=v.upper_bound, step=v.step)

        df = pd.DataFrame(list(product(*_iterdict.values())),
                          columns=_iterdict, dtype=object)[self.domain.names]

        return df


def grid(**kwargs) -> Sampler:
    return Grid(**kwargs)

# =============================================================================


class Sobol(Sampler):
    def __init__(self, n_samples, seed: Optional[int], **parameters):
        self.seed = seed
        self.parameters = parameters

    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        _continuous = sample_sobol_sequence(
            domain=self.domain.continuous, n_samples=n_samples)

        _discrete = sample_np_random_choice_range(
            domain=self.domain.discrete, n_samples=n_samples,
            seed=self.seed)

        _categorical = sample_np_random_choice(
            domain=self.domain.categorical, n_samples=n_samples,
            seed=self.seed)

        _constant = sample_constant(
            domain=self.domain.constant, n_samples=n_samples)

        df = pd.concat(
            [pd.DataFrame(_continuous, columns=self.domain.continuous.names),
             pd.DataFrame(_discrete, columns=self.domain.discrete.names),
             pd.DataFrame(
                _categorical, columns=self.domain.categorical.names),
             pd.DataFrame(_constant, columns=self.domain.constant.names)],
            axis=1
        )[self.domain.names]

        return df


def sobol(seed: Optional[int] = None, **kwargs) -> Sampler:
    return Sobol(seed=seed, **kwargs)


# =============================================================================

class Latin(Sampler):
    def __init__(self, seed: Optional[int], **parameters):
        self.seed = seed
        self.parameters = parameters

    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        _continuous = sample_latin_hypercube(
            domain=self.domain.continuous, n_samples=n_samples,
            seed=self.seed)

        _discrete = sample_np_random_choice_range(
            domain=self.domain.discrete, n_samples=n_samples,
            seed=self.seed)

        _categorical = sample_np_random_choice(
            domain=self.domain.categorical, n_samples=n_samples,
            seed=self.seed)

        _constant = sample_constant(
            domain=self.domain.constant, n_samples=n_samples)

        df = pd.concat(
            [pd.DataFrame(_continuous, columns=self.domain.continuous.names),
             pd.DataFrame(_discrete, columns=self.domain.discrete.names),
             pd.DataFrame(
                _categorical, columns=self.domain.categorical.names),
             pd.DataFrame(_constant, columns=self.domain.constant.names)],
            axis=1
        )[self.domain.names]

        return df


def latin(seed: Optional[int] = None, **kwargs) -> Sampler:
    return Latin(seed=seed, **kwargs)

# =============================================================================


_SAMPLERS = [random, latin, sobol, grid]

SAMPLER_MAPPING: Dict[str, Sampler] = {
    sampler.__name__.lower(): sampler for sampler in _SAMPLERS}

#                                                              Factory function
# =============================================================================


def _sampler_factory(sampler: str | Sampler, **parameters) -> Sampler:
    """
    Factory function for samplers

    Parameters
    ----------
    sampler : str | Sampler
        name of the sampler

    Returns
    -------
    Sampler
        sampler object
    """

    if isinstance(sampler, Sampler):
        return sampler

    elif isinstance(sampler, str):
        filtered_name = sampler.lower().replace(
            ' ', '').replace('-', '').replace('_', '')

        if filtered_name in SAMPLER_MAPPING:
            return SAMPLER_MAPPING[filtered_name](**parameters)

        else:
            raise KeyError(f"Unknown sampler name: {sampler}")

    else:
        raise TypeError(f"Unknown sampler type: {type(sampler)}")


SamplerNames = Literal['random', 'latin', 'sobol', 'grid']
