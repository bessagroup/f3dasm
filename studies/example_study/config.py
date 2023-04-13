from dataclasses import dataclass


@dataclass
class DesignConfig:
    number_of_samples: int

@dataclass
class ExperimentConfig:
    seed: int

@dataclass
class HPCConfig:
    jobid: int

@dataclass
class FileConfig:
    experimentdata_filename: str
    jobqueue_filename: str

@dataclass
class Config:
    design: DesignConfig
    experiment: ExperimentConfig
    hpc: HPCConfig
    file: FileConfig
