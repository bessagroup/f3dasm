Experiments
===========

This folder denotes experiments.

## Folder structure and files


```
├── experiments
│   └── my_experiment
│       ├── custom_module
│       │   ├── custom_script.py
│       │   └── __init__.py
│       ├── main.py
│       ├── config.py
│       ├── config.yaml
│       ├── default.yaml
│       ├── pbsjob.sh
│       └── README.md
└── src/f3dasm
```

* Each experiment is put in a separate folder, in this case `my_experiment`
* The README.md file gives a description, author and optionally citable source.
* The main script that has to be called should be name `main.py`
* Additional scripts or modules can be placed inside the `my_experiments` folder.
* `pbsjob.sh` is a [`TORQUE`](https://adaptivecomputing.com/cherry-services/torque-resource-manager/) file that will submit the `main.py` file to a high-performance queuing system.
* The `config.py`, `config.yaml` and `default.yaml` are [`hydra`](https://hydra.cc/docs/intro/) configuration files. More on that in the next section.

## Hydra

Configurations and data-storage for the experiments is handled by the [`hydra`](https://hydra.cc/docs/intro/) package.

* `config.py` denotes the types of all of the configurable parameters:

```python
from dataclasses import dataclass
from typing import Any, List

@dataclass
class SubConfig:
    parameter1: float
    parameter2: List[str]
    parameter3: int

@dataclass
class Config:
    subconfig: SubConfig
    parameter4: int
```

This will help you with type-hinting and write cleaner code.

* `config.yaml` is a [YAML](https://en.wikipedia.org/wiki/YAML) file containing the values of the configuration parameters:

```yaml
subconfig:
  parameter1: -1.0
  parameter2: ['banana','apple', 'pear']

parameter4: 3
```

* `default.yaml` contains the default values if we do not specify them in `config.yaml`, like we did with `parameter3` in the example:

```yaml
subconfig:
  parameter1: 0.0
  parameter2: ['fruit1','fruit2', 'fruit3']
  parameter3: 0

parameter4: 0
```

* A minimal `main.py` file will look something like this:


```python
import hydra
from config import Config
from hydra.core.config_store import ConfigStore

import f3dasm

@hydra.main(config_path=".", config_name="config")
def main(cfg: Config):
    ...


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

if __name__ == "__main__":
    main()

```

The configurations are given in the custom `Config` class type imported from `config.py` as input to the `main(cfg: Config)` function. This is done by the `@hydra.main()` decorater.

## Executing an experiment

Scripts can be run in two ways:

* Locally on your computer, by running the `main.py` file: 

```bash
$ python3 main.py
```

> Make sure you run the file in an environment where `f3dasm` and its dependencies are installed correctly!

* On a high-performance computer by submitting the `pbsjob.sh` to the queue:

```bash
$ qsub pbshjob.sh
```

> You can create array jobs easily in the commandline with the `-t` flag.

From the location that you executed/submitted the script, an `/outputs/` folder will be created, if not present.

In this `/outputs/` folder, a new folder will be created named:

* `/%year-%month-%day/%hour-%minute-%seconds/` locally
* `$PBS_JOBID/` on the HPC

The output-data, `hydra` configurations (`/.hyra/`) and logfile (`main.log`) will be automatically put in this folder

This will look something like this:


### Locally
```
├── outputs
    └── 2022-11-30
        └── 13-27-47
            ├── .hydra
            |   ├── config.yaml
            |   ├── hydra.yaml
            |   └── overrides.yaml
            ├── main.log
            └── data.obj
```

### HPC
```
├── outputs
    └── 448990
        ├── .hydra
        |   ├── config.yaml
        |   ├── hydra.yaml
        |   └── overrides.yaml
        ├── main.log
        └── data.obj
```