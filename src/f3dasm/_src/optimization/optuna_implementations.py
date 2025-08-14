#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Third-party
import optuna

# Locals
from ..core import Block
from ..datagenerator import DataGenerator
from ..design import Domain
from ..design.parameter import (
    CategoricalParameter,
    ConstantParameter,
    ContinuousParameter,
    DiscreteParameter,
)
from ..experimentdata import ExperimentData, ExperimentSample

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

# optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================


class OptunaOptimizer(Block):

    def __init__(self, data_generator: DataGenerator, algorithm_cls, seed: int, **hyperparameters):
        self.data_generator = data_generator
        self.algorithm_cls = algorithm_cls
        self.seed = seed
        self.hyperparameters = hyperparameters

    def arm(self, data: ExperimentData, output_name: str):

        self.output_name = output_name
        self.distributions = domain_to_optuna_distributions(
            data.domain)

        # Set algorithm
        self.study = optuna.create_study(
            sampler=self.algorithm_cls(seed=self.seed, **self.hyperparameters)
        )

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
        Perform a single step of the optimization process.
        This method is called by the `call` method.
        """
        trial = self.study.ask()
        new_es = self._suggest_experimentsample(
            trial=trial, domain=data.domain)

        new_experiment_data = ExperimentData.from_data(
            data={0: new_es},
            domain=data.domain,
            project_dir=data.project_dir)

        # Evaluate the sample with the data generator
        self.data_generator.arm(data=new_experiment_data)
        new_experiment_data = self.data_generator.call(
            data=new_experiment_data, **kwargs)

        self.study.add_trial(
            optuna.trial.create_trial(
                params=new_es.input_data,
                distributions=self.distributions,
                value=new_es.output_data[self.output_name],
            )
        )

        return new_experiment_data

    def call(self, data: ExperimentData, n_iterations: int, **kwargs) -> ExperimentData:
        for _ in range(n_iterations):
            data += self._step(data, **kwargs)

        return data
    def _suggest_experimentsample(self, trial: optuna.Trial, domain: Domain
                                  ) -> ExperimentSample:
        optuna_dict = {}
        for name, parameter in domain.input_space.items():
            if isinstance(parameter, ContinuousParameter):
                optuna_dict[name] = trial.suggest_float(
                    name=name,
                    low=parameter.lower_bound,
                    high=parameter.upper_bound, log=parameter.log)
            elif isinstance(parameter, DiscreteParameter):
                optuna_dict[name] = trial.suggest_int(
                    name=name,
                    low=parameter.lower_bound,
                    high=parameter.upper_bound, step=parameter.step)
            elif isinstance(parameter, CategoricalParameter):
                optuna_dict[name] = trial.suggest_categorical(
                    name=name,
                    choices=parameter.categories)
            elif isinstance(parameter, ConstantParameter):
                optuna_dict[name] = trial.suggest_categorical(
                    name=name, choices=[parameter.value])

        return ExperimentSample(input_data=optuna_dict,
                                domain=domain)


def domain_to_optuna_distributions(domain: Domain) -> dict:
    optuna_distributions = {}
    for name, parameter in domain.input_space.items():
        if parameter._type == 'float':
            optuna_distributions[
                name] = optuna.distributions.FloatDistribution(
                low=parameter.lower_bound,
                high=parameter.upper_bound, log=parameter.log)
        elif parameter._type == 'int':
            optuna_distributions[
                name] = optuna.distributions.IntDistribution(
                low=parameter.lower_bound,
                high=parameter.upper_bound, step=parameter.step)
        elif parameter._type == 'category':
            optuna_distributions[
                name] = optuna.distributions.CategoricalDistribution(
                parameter.categories)
        elif parameter._type == 'object':
            optuna_distributions[
                name] = optuna.distributions.CategoricalDistribution(
                choices=[parameter.value])
    return optuna_distributions
