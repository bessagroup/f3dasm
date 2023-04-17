from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DesignConfig:
    input_space: List[Dict]
    output_space: List[Dict]


@dataclass
class Config:
    design: DesignConfig
