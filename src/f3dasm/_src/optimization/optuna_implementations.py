#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Standard
from typing import Optional

# Third-party
import optuna

# Locals
from ..core import DataGenerator, Optimizer
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


class OptunaOptimizer(Optimizer):
    """
    Optuna-based optimizer block for experiment design and optimization.

    This class wraps Optuna's optimization logic and integrates it with the
    experiment data and domain definitions from f3dasm.
    """

    def __init__(self, optuna_sampler: optuna.samplers.BaseSampler):
        """
        Initialize the OptunaOptimizer.

        Parameters
        ----------
        optuna_sampler : optuna.samplers.BaseSampler
            The Optuna sampler to use for the optimization process.
        """
        self.optuna_sampler = optuna_sampler

    def arm(
        self,
        data: ExperimentData,
        data_generator: DataGenerator,
        output_name: str,
        input_name: Optional[str] = None,
    ):
        """
        Prepare the optimizer with experiment data, data generator,
        and output name.

        Parameters
        ----------
        data : ExperimentData
            The experiment data containing previous trials.
        data_generator : DataGenerator
            The data generator used to evaluate new samples.
        input_name : str
            The name of the input variable to optimize.
        output_name : str
            The name of the output variable to optimize.
        """
        self.data_generator = data_generator
        self.output_name = output_name
        self.distributions = domain_to_optuna_distributions(data.domain)

        # Set algorithm
        self.study = optuna.create_study(sampler=self.optuna_sampler)

        # Add existing trials to the study
        for _, es in data:
            self.study.add_trial(
                optuna.trial.create_trial(
                    params=es.input_data,
                    distributions=self.distributions,
                    value=es.output_data[self.output_name],
                )
            )

    def _step(self, data: ExperimentData, **kwargs) -> ExperimentData:
        """
        Perform a single optimization step.

        This method suggests a new experiment sample, evaluates it, and adds
        the result to the Optuna study.

        Parameters
        ----------
        data : ExperimentData
            The current experiment data.
        **kwargs
            Additional arguments passed to the data generator.

        Returns
        -------
        ExperimentData
            The updated experiment data including the new sample.
        """
        trial = self.study.ask()
        new_es = _suggest_experimentsample(trial=trial, domain=data.domain)

        new_experiment_data = ExperimentData.from_data(
            data={0: new_es}, domain=data.domain, project_dir=data._project_dir
        )

        # Evaluate the sample with the data generator
        self.data_generator.arm(data=new_experiment_data)
        new_experiment_data = self.data_generator.call(
            data=new_experiment_data, **kwargs
        )

        # TODO: extract last es
        new_es = new_experiment_data.get_experiment_sample(
            new_experiment_data.index[-1]
        )

        self.study.add_trial(
            optuna.trial.create_trial(
                params=new_es.input_data,
                distributions=self.distributions,
                value=new_es.output_data[self.output_name],
            )
        )

        return new_experiment_data

    def call(
        self, data: ExperimentData, n_iterations: int, **kwargs
    ) -> ExperimentData:
        """
        Run the optimization for a specified number of iterations.

        Parameters
        ----------
        data : ExperimentData
            The initial experiment data.
        n_iterations : int
            Number of optimization steps to perform.
        **kwargs
            Additional arguments passed to each optimization step.

        Returns
        -------
        ExperimentData
            The experiment data after optimization.
        """
        for _ in range(n_iterations):
            data += self._step(data, **kwargs)
        return data


def _suggest_experimentsample(
    trial: optuna.Trial, domain: Domain
) -> ExperimentSample:
    """
    Suggest a new experiment sample using Optuna trial and domain.

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
    """
    Convert a domain object to Optuna distributions.

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
    optuna_sampler: optuna.distributions.BaseDistribution,
) -> Optimizer:
    """Create an Optuna optimizer block.

    Parameters
    ----------
    optuna_sampler
        The Optuna sampler to use for the optimization.

    Returns
    -------
    Block
        An instance of the OptunaOptimizer block.
    """
    return OptunaOptimizer(optuna_sampler=optuna_sampler)


# =============================================================================


def tpesampler() -> Optimizer:
    """Create an Optuna TPE sampler optimizer block.

    Returns
    -------
    Block
        An instance of the OptunaOptimizer block with TPE sampler.
    """
    return OptunaOptimizer(optuna_sampler=optuna.samplers.TPESampler())
