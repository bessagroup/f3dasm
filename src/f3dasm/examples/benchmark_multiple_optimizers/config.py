from dataclasses import dataclass
from typing import Any, List


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
class OptimizersConfig:
    optimizers_names: List[str]


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
    optimizers: OptimizersConfig
    sampler: SamplerConfig
    execution: ExecutionConfig
