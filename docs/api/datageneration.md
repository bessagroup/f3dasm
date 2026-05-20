# Data-driven process

`Block` is the base class for all computational units in `f3dasm`. Samplers, data generators, and optimizer update steps are all plain `Block` subclasses; they compose with the `>>` operator into a `ChainedBlock` and can be repeated with `.loop(n)` to produce a `LoopBlock`. Use the `@datagenerator` decorator to quickly create blocks from functions.

!!! tip "See also"
    - Tutorial: [Understanding Blocks](../notebooks/data-driven/blocks.ipynb)
    - Tutorial: [Example: Car Stopping Distance](../notebooks/data-driven/carstoppingdistance.ipynb)
    - [Core Concepts: Block](../concepts.md#block)

::: f3dasm.Block

::: f3dasm.ChainedBlock

::: f3dasm.LoopBlock

::: f3dasm.DataGenerator

::: f3dasm.datagenerator