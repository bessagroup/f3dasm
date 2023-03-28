from dataclasses import dataclass


@dataclass
class FileConfig:
    recommender: str
    raw: str
    outputfilename: str


@dataclass
class HPCConfig:
    jobid: int


@dataclass
class Config:
    files: FileConfig
    hpc: HPCConfig
