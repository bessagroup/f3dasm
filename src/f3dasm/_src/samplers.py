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
from .design.domain import Domain
from .experimentdata import Block, ExperimentData

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================

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
            samples[:, dim] * (param.upper_bound - param.lower_bound)
            + param.lower_bound
        )

        # If param.log is True, take the 10** of the samples
        if param.log:
            samples[:, dim] = 10 ** samples[:, dim]

    return samples


def sample_constant(domain: Domain, n_samples: int) -> np.ndarray:
    """
    Sample the constant parameters.

    Parameters
    ----------
    domain : Domain
        The domain object containing the input space.
    n_samples : int
        The number of samples to generate.

    Returns
    -------
    np.ndarray
        The sampled data.
    """
    samples = np.array([param.value for param in domain.input_space.values()])
    return np.tile(samples, (n_samples, 1))


def sample_np_random_choice(
    domain: Domain, n_samples: int, seed: Optional[int] = None, **kwargs
) -> np.ndarray:
    """
    Sample with numpy random choice.

    Parameters
    ----------
    domain : Domain
        The domain object containing the input space.
    n_samples : int
        The number of samples to generate.
    seed : Optional[int], optional
        The random seed, by default None
    **kwargs : dict
        Additional parameters for sampling.

    Returns
    -------
    np.ndarray
        The sampled data.
    """
    rng = np.random.default_rng(seed)
    samples = np.empty(shape=(n_samples, len(domain)), dtype=object)
    for dim, param in enumerate(domain.input_space.values()):
        samples[:, dim] = rng.choice(param.categories, size=n_samples)

    return samples


def sample_np_random_choice_range(
    domain: Domain, n_samples: int, seed: Optional[int] = None, **kwargs
) -> np.ndarray:
    """
    Sample with numpy random choice within a range of values.

    Parameters
    ----------
    domain : Domain
        The domain object containing the input space.
    n_samples : int
        The number of samples to generate.
    seed : Optional[int], optional
        The random seed, by default None
    **kwargs : dict
        Additional parameters for sampling.

    Returns
    -------
    np.ndarray
        The sampled data.
    """
    samples = np.empty(shape=(n_samples, len(domain)), dtype=np.int32)
    rng = np.random.default_rng(seed)
    for dim, param in enumerate(domain.input_space.values()):
        samples[:, dim] = rng.choice(
            range(param.lower_bound, param.upper_bound + 1, param.step),
            size=n_samples,
        )

    return samples


def sample_np_random_uniform(
    domain: Domain, n_samples: int, seed: Optional[int] = None, **kwargs
) -> np.ndarray:
    """
    Sample with numpy random uniform distribution.

    Parameters
    ----------
    domain : Domain
        The domain object containing the input space.
    n_samples : int
        The number of samples to generate.
    seed : Optional[int], optional
        The random seed, by default None
    **kwargs : dict
        Additional parameters for sampling.

    Returns
    -------
    np.ndarray
        The sampled data.
    """
    rng = np.random.default_rng(seed)
    samples = rng.uniform(low=0.0, high=1.0, size=(n_samples, len(domain)))

    # stretch samples
    samples = _stretch_samples(domain, samples)
    return samples


def sample_np_random_uniform_array(
    domain: Domain, n_samples: int, seed: Optional[int] = None, **kwargs
) -> np.ndarray:
    """
    Sample with numpy random uniform distribution for array parameters.

    Parameters
    ----------
    domain : Domain
        The domain object containing the input space.
    n_samples : int
        The number of samples to generate.
    seed : Optional[int], optional
        The random seed, by default None
    **kwargs : dict
        Additional parameters for sampling.

    Returns
    -------
    np.ndarray
        The sampled data.
    """
    samples = []
    rng = np.random.default_rng(seed)
    for _ in range(n_samples):
        s = {}
        for name, param in domain.input_space.items():
            sample = rng.uniform(low=0.0, high=1.0, size=param.shape)
            # stretch samples
            sample = (
                sample * (param.upper_bound - param.lower_bound)
                + param.lower_bound
            )
            s[name] = sample
        samples.append(s)

    return samples


def sample_latin_hypercube(
    domain: Domain, n_samples: int, seed: Optional[int] = None, **kwargs
) -> np.ndarray:
    """
    Sample with Latin Hypercube sampling.

    Parameters
    ----------
    domain : Domain
        The domain object containing the input space.
    n_samples : int
        The number of samples to generate.
    seed : Optional[int], optional
        The random seed, by default None
    **kwargs : dict
        Additional parameters for sampling.

    Returns
    -------
    np.ndarray
        The sampled data.
    """
    if len(domain) == 0:
        return np.empty((n_samples, 0))

    problem = {
        "num_vars": len(domain),
        "names": domain.input_names,
        "bounds": [
            [s.lower_bound, s.upper_bound] for s in domain.input_space.values()
        ],
    }

    samples = salib_latin.sample(problem=problem, N=n_samples, seed=seed)
    return samples


def sample_latin_hypercube_array(
    domain: Domain,
    n_samples: int,
    seed: Optional[int] = None,
    **kwargs,
) -> list[dict[str, np.ndarray]]:
    """
    Sample with Latin Hypercube sampling for array parameters.

    Parameters
    ----------
    domain : Domain
        The domain object containing the input space.
    n_samples : int
        The number of samples to generate.
    seed : Optional[int], optional
        The random seed, by default None
    **kwargs : dict
        Additional parameters for sampling.

    Returns
    -------
    list of dict[str, np.ndarray]
        A list of samples, where each sample is a dictionary
        mapping parameter names to arrays.
    """
    samples = []

    for name, param in domain.array.input_space.items():
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

    # combine into list of dicts
    samples_dict = []
    for i in range(n_samples):
        _s = {}
        for idx, name in enumerate(domain.array.input_names):
            _s[name] = samples[idx][i]
        samples_dict.append(_s)

    return samples_dict


def sample_sobol_sequence(
    domain: Domain, n_samples: int, dimensionality: int, **kwargs
) -> np.ndarray:
    """
    Sample with Sobol sequence sampling.

    Parameters
    ----------
    domain : Domain
        The domain object containing the input space.
    n_samples : int
        The number of samples to generate.
    dimensionality : int
        The dimensionality of the input space.
    **kwargs : dict
        Additional parameters for sampling.

    Returns
    -------
    np.ndarray
        The sampled data.
    """

    if len(domain) == 0:
        return np.empty((n_samples, 0))

    samples = sobol_sequence.sample(N=n_samples, D=dimensionality)

    # stretch samples
    samples = _stretch_samples(domain, samples)
    return samples


def next_power_of_two(x: int) -> int:
    return 1 if x <= 1 else 2 ** (x - 1).bit_length()


def sample_sobol_sequence_array(
    domain: Domain,
    n_samples: int,
    seed: Optional[int] = None,
    **kwargs,
) -> list[dict[str, np.ndarray]]:
    """
    Sample with Sobol Sequence sampling for array parameters.

    Parameters
    ----------
    domain : Domain
        The domain object containing the input space.
    n_samples : int
        The number of samples to generate.
    seed : Optional[int], optional
        The random seed, by default None
    **kwargs : dict
        Additional parameters for sampling.

    Returns
    -------
    list of dict[str, np.ndarray]
        A list of samples, where each sample is a dictionary
        mapping parameter names to arrays.
    """
    samples = []

    for name, param in domain.array.input_space.items():
        lb = np.ravel(param.lower_bound)
        ub = np.ravel(param.upper_bound)

        problem = {
            "num_vars": prod(param.shape),
            "names": [f"{name}{i}" for i in range(prod(param.shape))],
            "bounds": list(zip(lb, ub, strict=False)),
        }

        # N must be power of 2
        N = next_power_of_two(ceil(n_samples / (prod(param.shape) + 2)))

        s = salib_sobol.sample(
            problem=problem,
            N=N,
            seed=seed,
            calc_second_order=False,
        )

        s = s[:n_samples].reshape((n_samples,) + param.shape)
        samples.append(s)

    samples_dict = []
    for i in range(n_samples):
        _s = {}
        for idx, name in enumerate(domain.array.input_names):
            _s[name] = samples[idx][i]
        samples_dict.append(_s)

    return samples_dict


#                                                             Built-in samplers
# =============================================================================


class RandomUniform(Block):
    def __init__(self, seed: Optional[int], **parameters):
        """
        Initialize the RandomUniform sampler.

        Parameters
        ----------
        seed : Optional[int]
            The random seed.
        **parameters : dict
            Additional parameters for the sampler.
        """
        self.seed = seed
        self.parameters = parameters

    def call(
        self, data: ExperimentData, n_samples: int, **kwargs
    ) -> ExperimentData:
        """
        Sample data using the RandomUniform method.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        **kwargs : dict
            Additional parameters for sampling.

        Returns
        -------
        pd.DataFrame
            The sampled data.
        """
        _continuous = sample_np_random_uniform(
            domain=data.domain.continuous, n_samples=n_samples, seed=self.seed
        )

        data_continuous = ExperimentData(
            input_data=pd.DataFrame(
                _continuous, columns=data.domain.continuous.input_names
            ),
            domain=data.domain.continuous,
            project_dir=data._project_dir,
        )

        _discrete = sample_np_random_choice_range(
            domain=data.domain.discrete, n_samples=n_samples, seed=self.seed
        )

        data_discrete = ExperimentData(
            input_data=pd.DataFrame(
                _discrete, columns=data.domain.discrete.input_names
            ),
            domain=data.domain.discrete,
            project_dir=data._project_dir,
        )

        _categorical = sample_np_random_choice(
            domain=data.domain.categorical, n_samples=n_samples, seed=self.seed
        )

        data_categorical = ExperimentData(
            input_data=pd.DataFrame(
                _categorical, columns=data.domain.categorical.input_names
            ),
            domain=data.domain.categorical,
            project_dir=data._project_dir,
        )

        _constant = sample_constant(data.domain.constant, n_samples)

        data_constant = ExperimentData(
            input_data=pd.DataFrame(
                _constant, columns=data.domain.constant.input_names
            ),
            domain=data.domain.constant,
            project_dir=data._project_dir,
        )

        _array = sample_np_random_uniform_array(
            domain=data.domain.array, n_samples=n_samples, seed=self.seed
        )

        data_array = ExperimentData(
            input_data=_array,
            domain=data.domain.array,
            project_dir=data._project_dir,
        )

        d = ExperimentData(
            project_dir=data._project_dir, domain=data.domain._copy()
        )

        for _d in [
            data_continuous,
            data_discrete,
            data_categorical,
            data_constant,
            data_array,
        ]:
            d = d.join(_d)

        # TODO: do we need this ?
        d.store_objects()

        return d


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
        stepsize_continuous_parameters : Optional[Dict[str, float] | float]
            The step size for continuous parameters.
        **kwargs : dict
            Additional parameters for sampling.

        Returns
        -------
        pd.DataFrame
            The sampled data.
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

    **kwargs : dict
        Additional parameters for the sampler.

    Returns
    -------
    Block
        An Block instance of a grid sampler.
    """
    return Grid(**kwargs)


# =============================================================================


class Sobol(Block):
    def __init__(self, seed: Optional[int], **parameters):
        """
        Initialize the Sobol sampler.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        seed : Optional[int]
            The random seed.
        **parameters : dict
            Additional parameters for the sampler.
        """
        self.seed = seed
        self.parameters = parameters

    def call(
        self, data: ExperimentData, n_samples: int, **kwargs
    ) -> ExperimentData:
        """
        Sample data using the Sobol method.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        **kwargs : dict
            Additional parameters for sampling.

        Returns
        -------
        pd.DataFrame
            The sampled data.
        """
        _continuous = sample_sobol_sequence(
            domain=data.domain.continuous,
            n_samples=n_samples,
            dimensionality=len(data.domain.continuous),
        )

        data_continuous = ExperimentData(
            input_data=pd.DataFrame(
                _continuous, columns=data.domain.continuous.input_names
            ),
            domain=data.domain.continuous,
            project_dir=data._project_dir,
        )

        _discrete = sample_np_random_choice_range(
            domain=data.domain.discrete, n_samples=n_samples, seed=self.seed
        )

        data_discrete = ExperimentData(
            input_data=pd.DataFrame(
                _discrete, columns=data.domain.discrete.input_names
            ),
            domain=data.domain.discrete,
            project_dir=data._project_dir,
        )

        _categorical = sample_np_random_choice(
            domain=data.domain.categorical, n_samples=n_samples, seed=self.seed
        )

        data_categorical = ExperimentData(
            input_data=pd.DataFrame(
                _categorical, columns=data.domain.categorical.input_names
            ),
            domain=data.domain.categorical,
            project_dir=data._project_dir,
        )

        _constant = sample_constant(data.domain.constant, n_samples)

        data_constant = ExperimentData(
            input_data=pd.DataFrame(
                _constant, columns=data.domain.constant.input_names
            ),
            domain=data.domain.constant,
            project_dir=data._project_dir,
        )

        _array = sample_sobol_sequence_array(
            domain=data.domain.array, n_samples=n_samples, seed=self.seed
        )

        data_array = ExperimentData(
            input_data=_array,
            domain=data.domain.array,
            project_dir=data._project_dir,
        )

        d = ExperimentData(
            project_dir=data._project_dir, domain=data.domain._copy()
        )

        for _d in [
            data_continuous,
            data_discrete,
            data_categorical,
            data_constant,
            data_array,
        ]:
            d = d.join(_d)

        return d


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


# =============================================================================


class Latin(Block):
    def __init__(self, seed: Optional[int], **parameters):
        """
        Initialize the Latin sampler.

        Parameters
        ----------
        seed : Optional[int]
            The random seed.
        **parameters : dict
            Additional parameters for the sampler.
        """
        self.seed = seed
        self.parameters = parameters

    def call(
        self, data: ExperimentData, n_samples: int, **kwargs
    ) -> ExperimentData:
        """
        Sample data using the Latin Hypercube method.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        **kwargs : dict
            Additional parameters for sampling.

        Returns
        -------
        pd.DataFrame
            The sampled data.
        """
        _continuous = sample_latin_hypercube(
            domain=data.domain.continuous, n_samples=n_samples, seed=self.seed
        )

        data_continuous = ExperimentData(
            input_data=pd.DataFrame(
                _continuous, columns=data.domain.continuous.input_names
            ),
            domain=data.domain.continuous,
            project_dir=data._project_dir,
        )

        _discrete = sample_np_random_choice_range(
            domain=data.domain.discrete, n_samples=n_samples, seed=self.seed
        )

        data_discrete = ExperimentData(
            input_data=pd.DataFrame(
                _discrete, columns=data.domain.discrete.input_names
            ),
            domain=data.domain.discrete,
            project_dir=data._project_dir,
        )

        _categorical = sample_np_random_choice(
            domain=data.domain.categorical, n_samples=n_samples, seed=self.seed
        )

        data_categorical = ExperimentData(
            input_data=pd.DataFrame(
                _categorical, columns=data.domain.categorical.input_names
            ),
            domain=data.domain.categorical,
            project_dir=data._project_dir,
        )

        _constant = sample_constant(data.domain.constant, n_samples)

        data_constant = ExperimentData(
            input_data=pd.DataFrame(
                _constant, columns=data.domain.constant.input_names
            ),
            domain=data.domain.constant,
            project_dir=data._project_dir,
        )

        _array = sample_latin_hypercube_array(
            domain=data.domain.array, n_samples=n_samples, seed=self.seed
        )

        data_array = ExperimentData(
            input_data=_array,
            domain=data.domain.array,
            project_dir=data._project_dir,
        )

        d = ExperimentData(
            project_dir=data._project_dir, domain=data.domain._copy()
        )

        for _d in [
            data_continuous,
            data_discrete,
            data_categorical,
            data_constant,
            data_array,
        ]:
            d = d.join(_d)

        return d


def latin(seed: Optional[int] = None, **kwargs) -> Block:
    """
    Create a lating hypercube sampler.

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
