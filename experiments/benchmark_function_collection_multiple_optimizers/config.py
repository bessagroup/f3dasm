from dataclasses import dataclass
from typing import Any, List


@dataclass
class DesignConfig:
    lower_bound: float
    upper_bound: float
    dimensionality_lower_bound: int
    dimensionality_upper_bound: int


@dataclass
class FunctionConfig:
    noise: float


@dataclass
class OptimizersConfig:
    optimizers_names: List[str]


@dataclass
class SamplerConfig:
    sampler_name: str
    number_of_samples: int


@dataclass
class ExecutionConfig:
    number_of_functions: int
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
