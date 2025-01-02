"""
Module containing the protocol classes to accommodate the Optimizer class
"""

#                                                                       Modules
# =============================================================================


from __future__ import annotations

# standard
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

# Third-party
import numpy as np
import pandas as pd

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


class Sampler(Protocol):
    ...

# =============================================================================


class Domain(Protocol):
    @property
    def continuous(self):
        ...

    def get_bounds(self) -> np.ndarray:
        ...

# =============================================================================


class DataGenerator(Protocol):
    def dfdx(self, x: np.ndarray) -> np.ndarray:
        ...

    def _run(
            self,
            experiment_sample: ExperimentSample | np.ndarray | Dict) -> \
            ExperimentSample:
        ...

# =============================================================================


class ExperimentSample(Protocol):
    ...

# =============================================================================


class ExperimentData(Protocol):

    def __init__(self, domain: Domain, input_data: np.ndarray,
                 output_data: np.ndarray):
        ...

    @property
    def domain(self) -> Domain:
        ...

    @classmethod
    def from_sampling(cls, domain: Domain, sampler: Sampler,
                      n_samples: int, seed: int) -> ExperimentData:
        ...

    def sample(self, sampler: Sampler, **kwargs):
        ...

    def evaluate(self, data_generator: DataGenerator, mode:
                 str, output_names: Optional[List[str]] = None, **kwargs):
        ...

    @property
    def index(self) -> pd.Index:
        ...

    def get_n_best_output(self, n_samples: int) -> ExperimentData:
        ...

    def to_numpy() -> Tuple[np.ndarray, np.ndarray]:
        ...

    def select(self, indices: int | slice | Iterable[int]) -> ExperimentData:
        ...

    def get_experiment_sample(self, id: int) -> ExperimentData:
        ...

    def remove_rows_bottom(self, n_rows: int):
        ...

    def add_experiments(self, experiment_sample: ExperimentData):
        ...

    def _overwrite_experiments(self, experiment_sample: ExperimentData,
                               indices: pd.Index, add_if_not_exist: bool):
        ...

    def _reset_index(self):
        ...
