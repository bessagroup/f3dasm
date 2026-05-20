# Core Concepts

This page explains the main building blocks of `f3dasm` and how they fit together.

---

## Overview

The `f3dasm` workflow revolves around four core abstractions:

```
Domain ──> ExperimentData ──> Block(s) ──> Pipeline
  │              │                │             │
  │  defines     │  stores        │  processes  │  chains blocks
  │  parameters  │  experiments   │  data       │  into workflows
```

1. A **Domain** defines the parameter space for your experiments.
2. An **ExperimentData** object holds all your experimental data (inputs and outputs).
3. **Blocks** are units of computation (sampling, data generation, optimization) that operate on `ExperimentData`.
4. A **Pipeline** chains multiple Blocks together into an automated workflow.

---

## Domain

The `Domain` defines what parameters your experiment has and their valid ranges. It supports several parameter types:

- **Continuous** (`add_float`) — floating-point values with lower and upper bounds
- **Discrete** (`add_int`) — integer values with lower and upper bounds
- **Categorical** (`add_category`) — a set of named categories
- **Constant** (`add_constant`) — a fixed value
- **Array** (`add_array`) — fixed-size numeric arrays

```python
from f3dasm.design import Domain

domain = Domain()
domain.add_float(name='temperature', low=200.0, high=800.0)
domain.add_int(name='num_layers', low=1, high=10)
domain.add_category(name='material', categories=['steel', 'aluminum', 'titanium'])
```

**Learn more:** [Creating a Domain](notebooks/design/domain_creation.ipynb) | [API Reference](api/design.md)

---

## ExperimentData

`ExperimentData` is the central data container. It stores both the input parameters and the output results of your experiments in a structured way. You can think of it as a smart table where each row is one experiment.

```python
from f3dasm import ExperimentData

data = ExperimentData(domain=domain)
```

Key capabilities:

- Store experiments to disk and reload them
- Export to pandas DataFrames
- Track experiment status (open, in progress, finished, error)
- Access individual experiments via `ExperimentSample`

**Learn more:** [Working with ExperimentData](notebooks/experimentdata/experimentdata.ipynb) | [API Reference](api/experimentdata.md)

---

## Block

A `Block` is the fundamental unit of computation in `f3dasm`. Every operation on your data — sampling, evaluation, optimization — is a Block with a uniform signature: `call(data: ExperimentData, **kwargs) -> ExperimentData`.

Blocks compose with the `>>` operator into a `ChainedBlock`, and can be repeated with `.loop(n)` to produce a `LoopBlock`. Samplers, data generators, and optimizer update steps are all just Blocks — there's no separate `Sampler` or `Optimizer` class hierarchy.

Built-in Blocks:

| Category | Examples |
|----------|---------|
| Samplers | `RandomUniform`, `Latin`, `Sobol`, `Grid` (and the factories `random`, `latin`, `sobol`, `grid`) |
| Data generators | `DataGenerator` subclasses or the `@datagenerator` decorator |
| Optimizer update steps | `tpesampler(output_name=...)` (ask/tell style) |
| One-shot optimizers | `cg(...)`, `lbfgsb(...)`, `nelder_mead(...)` (scipy's own inner loop) |

You can create your own Block by subclassing:

```python
from f3dasm import Block, ExperimentData

class MyBlock(Block):
    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        # transform data here
        return data
```

### Composing Blocks

Each Block's `call` accepts the same `**kwargs`, which are passed to every block in a chain. Keep sampling and evaluation separate when they need different call-time arguments, and use `>>` to chain blocks whose call-time arguments are compatible:

```python
from f3dasm import create_optimizer, create_sampler

# Initial design: sample, then evaluate
data = create_sampler("latin", seed=42).call(data, n_samples=20)
data = data_generator.call(data)

# Optimize: chain the ask/tell update step with the data generator and
# wrap the pair in a LoopBlock. create_optimizer returns just the update
# step for ask/tell optimizers (like "tpesampler").
update_step = create_optimizer("tpesampler", output_name="y")
loop = (update_step >> data_generator).loop(50)
loop.arm(data)
data = loop.call(data)
```

**Learn more:** [Understanding Blocks](notebooks/data-driven/blocks.ipynb) | [Built-in Defaults](defaults.md) | [API Reference](api/datageneration.md)

---

## Pipeline

A `Pipeline` chains Blocks together into a complete workflow. It handles the execution order, data passing between steps, and supports both local execution and HPC clusters via SLURM.

```python
from f3dasm import Pipeline, Step, Loop

pipeline = Pipeline(
    steps=[
        Step(block=sampler),
        Loop(
            steps=[
                Step(block=evaluator),
                Step(block=optimizer),
            ],
            iterations=10,
        ),
    ]
)

pipeline.execute(data)
```

Key features:

- **Sequential and looped execution** — chain steps or repeat them
- **SLURM support** — run on HPC clusters with `SlurmCluster`
- **Resumable** — pipelines can be interrupted and resumed
- **Import path preservation** — local Python imports (`from my_script import func`) are automatically resolved on SLURM compute nodes via shared filesystem

**Learn more:** [Building a Pipeline](notebooks/pipeline/pipeline.ipynb) | [API Reference](api/pipeline.md)

---

## How It All Fits Together

A typical `f3dasm` workflow follows this pattern:

1. **Define** your parameter space with a `Domain`
2. **Create** an `ExperimentData` object from the domain
3. **Sample** initial experiments using a sampler Block
4. **Evaluate** the experiments using a data generation Block
5. **Optimize** by iterating between evaluation and optimization Blocks
6. **Automate** the whole process with a `Pipeline`

For a hands-on walkthrough, see the [tutorials section](notebooks/design/domain_creation.ipynb) or jump straight to the [quickstart notebook](notebooks/quickstart.ipynb).
