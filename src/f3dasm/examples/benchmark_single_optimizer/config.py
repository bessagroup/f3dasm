from dataclasses import dataclass
from typing import Any


@dataclass
class DesignConfig:
    lower_bound: float
    upper_bound: float
    dimensionality: int


@dataclass
class FunctionConfig:
    function_name: str
    noise: float or Any


@dataclass
class OptimizerConfig:
    optimizer_name: str
    hyperparameters: dict


@dataclass
class SamplerConfig:
    sampler_name: str
    number_of_samples: int


@dataclass
class ExecutionConfig:
    iterations: int
    realizations: int
    parallelization: bool


@dataclass
class Config:
    design: DesignConfig
    function: FunctionConfig
    optimizer: OptimizerConfig
    sampler: SamplerConfig
    execution: ExecutionConfig
