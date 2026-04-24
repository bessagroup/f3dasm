#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Standard
from typing import Optional

# Third-party
import optuna

# Locals
from ..core import Block
from ..design import Domain
from ..design.parameter import (
    ArrayParameter,
    CategoricalParameter,
    ConstantParameter,
    ContinuousParameter,
    DiscreteParameter,
)
from ..experimentdata import ExperimentData, ExperimentSample

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


class OptunaUpdateStep(Block):
    """Single-step optimizer update driven by an Optuna sampler.

    One ``call`` performs one ask/append cycle: it registers the previous
    iteration's evaluated candidate with the Optuna study (if any), asks
    the sampler for a new candidate, and appends it as an unevaluated row
    to the returned :class:`ExperimentData`. Wrap in a :class:`LoopBlock`
    and chain with a :class:`DataGenerator` to drive an optimization loop:

    >>> step = tpesampler(output_name="y") >> data_generator
    >>> data = step.loop(50).call(initial_data)

    Parameters
    ----------
    optuna_sampler : optuna.samplers.BaseSampler
        The Optuna sampler used for suggestion.
    output_name : str
        Name of the output column to minimize.

    Attributes
    ----------
    optuna_sampler : optuna.samplers.BaseSampler
        The configured Optuna sampler.
    output_name : str
        Name of the output column being minimized.
    study : optuna.study.Study
        The Optuna study (created in :meth:`arm`).
    distributions : dict
        Optuna distributions derived from the domain (set in :meth:`arm`).
    """

    def __init__(
        self,
        optuna_sampler: optuna.samplers.BaseSampler,
        output_name: str,
    ):
        self.optuna_sampler = optuna_sampler
        self.output_name = output_name

    def arm(self, data: ExperimentData) -> None:
        """Create the Optuna study and seed it with ``data``'s history.

        Parameters
        ----------
        data : ExperimentData
            Experiment data whose already-evaluated rows are registered
            with the new study as historical trials.
        """
        self.distributions = domain_to_optuna_distributions(data.domain)
        self.study = optuna.create_study(sampler=self.optuna_sampler)

        for _, es in data:
            self.study.add_trial(
                optuna.trial.create_trial(
                    params=es.input_data,
                    distributions=self.distributions,
                    value=es.output_data[self.output_name],
                )
            )

        self._pending_trial_index: Optional[int] = None

    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        """Register the previous iteration's result, ask for one new
        candidate, and return ``data`` with the candidate appended.

        Parameters
        ----------
        data : ExperimentData
            The experiment data. The candidate from the previous
            iteration (if any) must already be evaluated.
        **kwargs : dict
            Ignored; accepted for chaining compatibility.

        Returns
        -------
        ExperimentData
            ``data`` with one new unevaluated row appended.
        """
        if self._pending_trial_index is not None:
            es = data.get_experiment_sample(self._pending_trial_index)
            self.study.add_trial(
                optuna.trial.create_trial(
                    params=es.input_data,
                    distributions=self.distributions,
                    value=es.output_data[self.output_name],
                )
            )
            self._pending_trial_index = None

        trial = self.study.ask()
        new_es = _suggest_experimentsample(trial=trial, domain=data.domain)

        new_data = ExperimentData.from_data(
            data={0: new_es},
            domain=data.domain,
            project_dir=data._project_dir,
        )
        merged = data + new_data
        self._pending_trial_index = merged.index[-1]
        return merged


def _suggest_experimentsample(
    trial: optuna.Trial, domain: Domain
) -> ExperimentSample:
    """Suggest a new experiment sample using an Optuna trial and domain.

    Parameters
    ----------
    trial : optuna.Trial
        The Optuna trial object for suggesting parameters.
    domain : Domain
        The domain describing the input space.

    Returns
    -------
    ExperimentSample
        The suggested experiment sample.
    """
    optuna_dict = {}
    for name, parameter in domain.input_space.items():
        if isinstance(parameter, ContinuousParameter):
            optuna_dict[name] = trial.suggest_float(
                name=name,
                low=parameter.lower_bound,
                high=parameter.upper_bound,
                log=parameter.log,
            )
        elif isinstance(parameter, DiscreteParameter):
            optuna_dict[name] = trial.suggest_int(
                name=name,
                low=parameter.lower_bound,
                high=parameter.upper_bound,
                step=parameter.step,
            )
        elif isinstance(parameter, CategoricalParameter):
            optuna_dict[name] = trial.suggest_categorical(
                name=name, choices=parameter.categories
            )
        elif isinstance(parameter, ConstantParameter):
            optuna_dict[name] = trial.suggest_categorical(
                name=name, choices=[parameter.value]
            )
        elif isinstance(parameter, ArrayParameter):
            raise ValueError(
                "ArrayParameter is not supported in Optuna trials. "
                "Please use a different parameter types in your Domain."
            )
        else:
            raise TypeError(
                f"Unsupported parameter type: {type(parameter)} for {name}"
            )
    return ExperimentSample(_input_data=optuna_dict)


def domain_to_optuna_distributions(domain: Domain) -> dict:
    """Convert a domain to Optuna distributions.

    Parameters
    ----------
    domain : Domain
        The domain describing the input space.

    Returns
    -------
    dict
        Dictionary mapping parameter names to Optuna distributions.
    """
    optuna_distributions = {}
    for name, parameter in domain.input_space.items():
        if isinstance(parameter, ContinuousParameter):
            optuna_distributions[name] = (
                optuna.distributions.FloatDistribution(
                    low=parameter.lower_bound,
                    high=parameter.upper_bound,
                    log=parameter.log,
                )
            )
        elif isinstance(parameter, DiscreteParameter):
            optuna_distributions[name] = optuna.distributions.IntDistribution(
                low=parameter.lower_bound,
                high=parameter.upper_bound,
                step=parameter.step,
            )
        elif isinstance(parameter, CategoricalParameter):
            optuna_distributions[name] = (
                optuna.distributions.CategoricalDistribution(
                    parameter.categories
                )
            )
        elif isinstance(parameter, ConstantParameter):
            optuna_distributions[name] = (
                optuna.distributions.CategoricalDistribution(
                    choices=[parameter.value]
                )
            )
        elif isinstance(parameter, ArrayParameter):
            raise ValueError(
                "ArrayParameter is not supported in Optuna trials. "
                "Please use a different parameter types in your Domain."
            )
        else:
            raise TypeError(
                f"Unsupported parameter type: {type(parameter)} for {name}"
            )
    return optuna_distributions


def optuna_optimizer(
    optuna_sampler: optuna.samplers.BaseSampler,
    output_name: str,
) -> OptunaUpdateStep:
    """Create an Optuna-backed update-step block.

    Parameters
    ----------
    optuna_sampler : optuna.samplers.BaseSampler
        The Optuna sampler to use.
    output_name : str
        Name of the output column to minimize.

    Returns
    -------
    OptunaUpdateStep
        Configured update-step block.
    """
    return OptunaUpdateStep(
        optuna_sampler=optuna_sampler, output_name=output_name
    )


# =============================================================================


def tpesampler(output_name: str) -> OptunaUpdateStep:
    """Create an Optuna TPE-sampler update-step block.

    Parameters
    ----------
    output_name : str
        Name of the output column to minimize.

    Returns
    -------
    OptunaUpdateStep
        Update-step block wrapping Optuna's TPE sampler.
    """
    return OptunaUpdateStep(
        optuna_sampler=optuna.samplers.TPESampler(),
        output_name=output_name,
    )
