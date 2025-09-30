# Built-in functionalities

`f3dasm` provides a set of built-in functionalities that can be used to perform data-driven optimization and sensitivity analysis.  
All built-ins are implementations of the `Block` class and can be used on your `ExperimentData` object.

The built-in blocks can be initialized by either importing the functions directly from the respective submodules or by using a string argument to specify the built-in function you want to use.

| Part of the data-driven process | Submodule for built-ins       | Function to call with string argument |
|--------------------------------|-------------------------------|--------------------------------------|
| Sampling                        | `f3dasm.design`              | `f3dasm.create_sampler`              |
| Data generation                 | `f3dasm.datageneration`      | `f3dasm.create_datagenerator`        |
| Optimization                    | `f3dasm.optimization`        | `f3dasm.create_optimizer`            |

`f3dasm` provides two ways to use the built-in functionalities:

## 1. Call the built-in functions

You can import the built-in functions directly from the respective submodules and call them to change the (hyper)parameters.

```python
from f3dasm.design import random
from f3dasm.datageneration import ackley

# Call the random uniform sampler with a specific seed
sampler_block = random(seed=123)

# Create a 2D instance of the 'Ackley' function with its box-constraints scaled to [0, 1]
data_generation_block = ackley(scale_bounds=[[0., 1.], [0., 1.]])

# Create an empty Domain
domain = Domain()

# Add two continuous parameters 'x0' and 'x1'
domain.add_float(name='x0', low=0.0, high=1.0)
domain.add_float(name='x1', low=0.0, high=1.0)

# Create an empty ExperimentData object with the domain
experiment_data = ExperimentData(domain=domain)

# 1. Sampling
experiment_data = sampler_block(data=experiment_data, n_samples=10)

# 2. Evaluating the samples
data_generation_block.arm(data=experiment_data)
experiment_data = data_generation_block.call(data=experiment_data)
```

## 2. Use a string argument

Alternatively, you can use a string argument to specify the built-in function you want to use.

```python
from f3dasm import create_sampler, create_datagenerator

sampler_block = create_sampler(sampler='random', seed=123)

data_generation_block = create_datagenerator(
    data_generator='ackley',
    scale_bounds=[[0., 1.], [0., 1.]]
)

# Create an empty Domain
domain = Domain()
domain.add_float(name='x0', low=0.0, high=1.0)
domain.add_float(name='x1', low=0.0, high=1.0)

# Create an empty ExperimentData object with the domain
experiment_data = ExperimentData(domain=domain)

# 1. Sampling
experiment_data = sampler_block(data=experiment_data, n_samples=10)

# 2. Evaluating the samples
data_generation_block.arm(data=experiment_data)
experiment_data = data_generation_block.call(data=experiment_data)
```

## Implemented samples

The following built-in implementations of samplers can be used in the data-driven process.

| Name                     | Key-word   | Function               | Reference                                                                                                               |
| ------------------------ | ---------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Random Uniform sampling  | `"random"` | `f3dasm.design.random` | [numpy.random.uniform](https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html)               |
| Latin Hypercube sampling | `"latin"`  | `f3dasm.design.latin`  | [SALib.latin](https://salib.readthedocs.io/en/latest/api/SALib.sample.html#SALib.sample.latin.sample)                   |
| Sobol Sequence sampling  | `"sobol"`  | `f3dasm.design.sobol`  | [SALib.sobol_sequence](https://salib.readthedocs.io/en/latest/api/SALib.sample.html#SALib.sample.sobol_sequence.sample) |
| Grid Search sampling     | `"grid"`   | `f3dasm.design.grid`   | [itertools.product](https://docs.python.org/3/library/itertools.html#itertools.product)                                 |

## Implemented benchmark functions

### Convex functions

| Name                    | Key-word                    | Function                                                |
| ----------------------- | --------------------------- | ------------------------------------------------------- |
| Ackley N. 2             | `"Ackley N. 2"`             | `f3dasm.datageneration.functions.ackleyn2`              |
| Bohachevsky N. 1        | `"Bohachevsky N. 1"`        | `f3dasm.datageneration.functions.bohachevskyn1`         |
| Booth                   | `"Booth"`                   | `f3dasm.datageneration.functions.booth`                 |
| Brent                   | `"Brent"`                   | `f3dasm.datageneration.functions.brent`                 |
| Brown                   | `"Brown"`                   | `f3dasm.datageneration.functions.brown`                 |
| Bukin N. 6              | `"Bukin N. 6"`              | `f3dasm.datageneration.functions.bukinn6`               |
| Dixon Price             | `"Dixon Price"`             | `f3dasm.datageneration.functions.dixonprice`            |
| Exponential             | `"Exponential"`             | `f3dasm.datageneration.functions.exponential`           |
| Matyas                  | `"Matyas"`                  | `f3dasm.datageneration.functions.matyas`                |
| McCormick               | `"McCormick"`               | `f3dasm.datageneration.functions.mccormick`             |
| Powell                  | `"Powell"`                  | `f3dasm.datageneration.functions.powell`                |
| Rotated Hyper-Ellipsoid | `"Rotated Hyper-Ellipsoid"` | `f3dasm.datageneration.functions.rotatedhyperellipsoid` |
| Schwefel 2.20           | `"Schwefel 2.20"`           | `f3dasm.datageneration.functions.schwefel2_20`          |
| Schwefel 2.21           | `"Schwefel 2.21"`           | `f3dasm.datageneration.functions.schwefel2_21`          |
| Schwefel 2.22           | `"Schwefel 2.22"`           | `f3dasm.datageneration.functions.schwefel2_22`          |
| Schwefel 2.23           | `"Schwefel 2.23"`           | `f3dasm.datageneration.functions.schwefel2_23`          |
| Sphere                  | `"Sphere"`                  | `f3dasm.datageneration.functions.sphere`                |
| Sum Squares             | `"Sum Squares"`             | `f3dasm.datageneration.functions.sumsquares`            |
| Thevenot                | `"Thevenot"`                | `f3dasm.datageneration.functions.thevenot`              |
| Trid                    | `"Trid"`                    | `f3dasm.datageneration.functions.trid`                  |

### Seperable functions

| Name             | Key-word             | Function                                         |
| ---------------- | -------------------- | ------------------------------------------------ |
| Ackley           | `"Ackley"`           | `f3dasm.datageneration.functions.ackley`         |
| Bohachevsky N. 1 | `"Bohachevsky N. 1"` | `f3dasm.datageneration.functions.bohachevskyn1`  |
| Easom            | `"Easom"`            | `f3dasm.datageneration.functions.easom`          |
| Egg Crate        | `"Egg Crate"`        | `f3dasm.datageneration.functions.eggcrate`       |
| Exponential      | `"Exponential"`      | `f3dasm.datageneration.functions.exponential`    |
| Griewank         | `"Griewank"`         | `f3dasm.datageneration.functions.griewank`       |
| Michalewicz      | `"Michalewicz"`      | `f3dasm.datageneration.functions.michalewicz`    |
| Powell           | `"Powell"`           | `f3dasm.datageneration.functions.powell`         |
| Qing             | `"Qing"`             | `f3dasm.datageneration.functions.qing`           |
| Quartic          | `"Quartic"`          | `f3dasm.datageneration.functions.quartic`        |
| Rastrigin        | `"Rastrigin"`        | `f3dasm.datageneration.functions.rastrigin`      |
| Schwefel         | `"Schwefel"`         | `f3dasm.datageneration.functions.schwefel`       |
| Schwefel 2.20    | `"Schwefel 2.20"`    | `f3dasm.datageneration.functions.schwefel2_20`   |
| Schwefel 2.21    | `"Schwefel 2.21"`    | `f3dasm.datageneration.functions.schwefel2_21`   |
| Schwefel 2.22    | `"Schwefel 2.22"`    | `f3dasm.datageneration.functions.schwefel2_22`   |
| Schwefel 2.23    | `"Schwefel 2.23"`    | `f3dasm.datageneration.functions.schwefel2_23`   |
| Sphere           | `"Sphere"`           | `f3dasm.datageneration.functions.sphere`         |
| Styblinski Tank  | `"Styblinski Tank"`  | `f3dasm.datageneration.functions.styblinskitang` |
| Sum Squares      | `"Sum Squares"`      | `f3dasm.datageneration.functions.sumsquares`     |
| Thevenot         | `"Thevenot"`         | `f3dasm.datageneration.functions.thevenot`       |
| Xin She Yang     | `"Xin She Yang"`     | `f3dasm.datageneration.functions.xin_she_yang`   |


### Multimodal functions

| Name             | Key-word             | Function                                        |
| ---------------- | -------------------- | ----------------------------------------------- |
| Ackley           | `"Ackley"`           | `f3dasm.datageneration.functions.ackley`        |
| Ackley N. 3      | `"Ackley N. 3"`      | `f3dasm.datageneration.functions.ackleyn3`      |
| Ackley N. 4      | `"Ackley N. 4"`      | `f3dasm.datageneration.functions.ackleyn4`      |
| Adjiman          | `"Adjiman"`          | `f3dasm.datageneration.functions.adjiman`       |
| Bartels          | `"Bartels"`          | `f3dasm.datageneration.functions.bartels`       |
| Beale            | `"Beale"`            | `f3dasm.datageneration.functions.beale`         |
| Bird             | `"Bird"`             | `f3dasm.datageneration.functions.bird`          |
| Bohachevsky N. 2 | `"Bohachevsky N. 2"` | `f3dasm.datageneration.functions.bohachevskyn2` |
| ...              | ...                  | ...                                             |

## Implemented optimizers

The following implementations of optimizers can be found under the f3dasm.optimization module.
These are ported from scipy-optimize

| Name               | Key-word          | Function                            | Reference                                                                                                 |
| ------------------ | ----------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Conjugate Gradient | `"cg"`            | `f3dasm.optimization.cg`            | [scipy.minimize CG](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cg.html)                 |
| L-BFGS-B           | `"lbfgsb"`        | `f3dasm.optimization.lbfgsb`        | [scipy.minimize L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)       |
| Nelder Mead        | `"nelder_mead"`   | `f3dasm.optimization.nelder_mead`   | [scipy.minimize NelderMead](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html) |
| Random search      | `"random_search"` | `f3dasm.optimization.random_search` | [numpy](https://numpy.org/doc/)                                                                           |
