Studies
=======

This folder denotes benchmark studies that can be run with the `f3dasm` package.
In order to run a study, you need to have the `f3dasm[benchmark]` extra requirements installed:

```
pip install f3dasm[benchmark]
```

## Folder structure and files of a study

```
├── .
│   └── my_study
│       ├── main.py
│       ├── config.yaml
│       ├── pbsjob.sh
│       └── README.md
└── src/f3dasm
```

* Each study is put in a separate folder
* The `README.md` file gives a description, author and optionally citable source.
* The main script that has to be called should be named `main.py`
* `pbsjob.sh` is a batchscript file that will submit the `main.py` file to a [`TORQUE`](https://adaptivecomputing.com/cherry-services/torque-resource-manager/) high-performance queuing system.
* The `config.yaml` are [`hydra`](https://hydra.cc/docs/intro/) configuration files.

## Available studies

There are two benchmark studies available:

| Study | Description | 
| :-- | :-- |
| Fragile becomes supercompressible | A benchmark study that compares the performance of the `f3dasm` package with other packages. |
| Comparing optimization algorithms on benchmark functions | A benchmark study that compares the performance of the `f3dasm` package with other packages. |