# f3dasm

The ubiquitous language of f3dasm — a framework for data-driven design and analysis of structures and materials. This file is a glossary, not a spec: it defines what each term *is*, free of implementation detail.

## The data model

**Domain**:
The declared input/output parameter space of a problem.
_Avoid_: search space, design space (when you mean the declared `Domain` object).

**Parameter**:
One named axis of a Domain — continuous, discrete, categorical, constant, or array-valued.
_Avoid_: variable, feature, dimension.

**ExperimentData**:
The central collection of experiment samples that flows between blocks and is persistable to disk. The value type the whole framework passes around.
_Avoid_: dataset, table, dataframe.

**ExperimentSample**:
A single row of ExperimentData: its input values, output values, and job status.
_Avoid_: row, record, datapoint, trial.

**Job status**:
The lifecycle state of one ExperimentSample's evaluation: open → in progress → finished, or error.
_Avoid_: state (alone), stage.

## Computation

**Block**:
A unit of computation that transforms ExperimentData through one uniform call. Sampling, optimisation update steps, and transforms are all blocks; there is deliberately no separate class hierarchy beneath it.
_Avoid_: operator, stage, transformer, task.

**DataGenerator**:
A per-sample evaluator — given an ExperimentSample's inputs it computes that sample's outputs. Drives the actual model/simulation evaluation.
_Avoid_: function (alone), evaluator, simulator.

**Parallelization mode**:
The strategy by which a DataGenerator evaluates the samples in an ExperimentData — sequentially, in parallel, on a cluster, or as a cluster array. A property of *how* evaluation happens, not *what* is evaluated.
_Avoid_: backend, strategy.

## Execution

**Pipeline**:
An ordered, resumable workflow of steps.
_Avoid_: workflow, DAG, job (alone).

**Step**:
One entry in a Pipeline: a block together with how it is run (its keyword arguments, whether it is parallel, where its data lives).
_Avoid_: stage, node, task.

**Executor**:
The backend that runs a Pipeline in a particular environment — in the current process, or by submitting jobs to a SLURM cluster.
_Avoid_: runner, scheduler, backend.

**Execution context**:
The description of *how and where* a single step is evaluated in a given environment — its parallelization mode and which jobs the current invocation is responsible for — independent of *what* the step computes. The same step runs locally or as one task of a SLURM array purely by changing its execution context.
_Avoid_: config, options, environment (alone).

**Wave**:
One SLURM array submission covering a contiguous window of a parallel step's open jobs. A parallel step executes as one or more waves, each gated on the previous wave terminating.
_Avoid_: batch, chunk, round.
