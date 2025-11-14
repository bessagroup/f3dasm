# Studies

[**TORQUE**](https://adaptivecomputing.com/cherry-services/torque-resource-manager/)  


To get a feeling for a data-driven experiment, two benchmark studies are available to run with the `f3dasm` package.  
In order to run a study, you need to have the `f3dasm[benchmark]` extra requirements installed:

```bash
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

- Each study is put in a separate folder.
- The `README.md` file gives a description, author, and optionally a citable source.
- The script that has to be called should be named `main.py`.
- *(optional)* `config.yaml` is a [**hydra**](https://hydra.cc/docs/intro/) configuration file that contains all the parameters for the experiment
- *(optional)* `pbsjob.sh` is a batch script file that will submit the `main.py` file to a HPC system

## Available studies

There are two benchmark studies available:

| Study                                                                                                                                     | Description                                                                                                                                                           |
| ----------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Fragile becomes supercompressible](https://github.com/bessagroup/f3dasm/tree/main/studies/fragile_becomes_supercompressible)             | Designing a supercompressible meta-material. This study focuses on creating a meta-material that exhibits supercompressibility under loading conditions.              |
| [Comparing optimization algorithms on benchmark functions](https://github.com/bessagroup/f3dasm/tree/pr/1.5/studies/benchmark_optimizers) | Benchmark various optimization algorithms on analytical functions. Includes tests with multiple objective functions and comparison metrics for optimizer performance. |
