from dataclasses import dataclass
from typing import Any, List


@dataclass
class DesignConfig:
    lower_bound: float
    upper_bound: float
    dimensionality: int


@dataclass
class FunctionConfig:
    fidelity_values: List[float]
    costs: List[float]
    noise: float or Any


@dataclass
class OptimizerConfig:
    optimizer_name: str


@dataclass
class SamplerConfig:
    sampler_name: str
    numbers_of_samples: List[int]


@dataclass
class ExecutionConfig:
    iterations: int
    realizations: int
    parallelization: bool
    seed: int


@dataclass
class Config:
    design: DesignConfig
    function: FunctionConfig
    optimizer: OptimizerConfig
    sampler: SamplerConfig
    execution: ExecutionConfig
