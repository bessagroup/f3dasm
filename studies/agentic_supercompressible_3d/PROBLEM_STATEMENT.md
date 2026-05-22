# Problem Briefing: Supercompressible Metamaterial Design (3D)

I am a researcher running an agentic optimization system against a structural
mechanics problem I have not solved before. This document is your complete
starting point. Read it carefully, then use Ask() to clarify any genuine
ambiguities before forming hypotheses.

---

## What I am trying to do

I have a conical deployable mast made of PLA (polylactic acid), a brittle
polymer. When compressed along its axis, most geometries cause the mast to
buckle and break. For a specific subset of the design space, compression
instead causes the mast to coil along its axis — a reversible mode that
allows it to spring back to its original shape after full compression. I want
to find the geometry that coils reversibly AND sustains the highest possible
critical buckling stress before instability onset. This is a constrained
optimization problem over a three-dimensional continuous design space.

---

## Design space

Three continuous input parameters define the geometry. Longerons have
**circular cross-sections** — only the diameter varies. All other geometric
and material parameters are fixed and must never be proposed.

| Parameter             | Bounds           | Physical meaning                                                              |
|-----------------------|------------------|-------------------------------------------------------------------------------|
| `ratio_d`             | [0.004, 0.073]   | Longeron cross-section diameter relative to bottom ring diameter: d / D1      |
| `ratio_pitch`         | [0.25, 1.50]     | Mast height relative to bottom diameter: P / D1                               |
| `ratio_top_diameter`  | [0.0, 0.8]       | Taper of the mast: (D1 − D2) / D1                                             |

Fixed constants (do not vary these):

| Constant               | Value     |
|------------------------|-----------|
| `young_modulus`        | 3500 MPa  |
| `n_longerons`          | 3         |
| `bottom_diameter`      | 100 mm    |
| `ratio_shear_modulus`  | 0.3677    |
| `circular`             | True      |

---

## Outputs and what they mean

Each evaluated design produces three output values:

**`coilable`** — integer with three possible values:
- `0` — the mast buckles in the wrong mode (global bending, local longeron
  buckling); it does not coil at all. Fundamental geometry failure.
- `1` — the mast coils reversibly when compressed; maximum local strain
  throughout full compression stays below 2%, so PLA does not fracture.
  This is the only acceptable outcome.
- `2` — the mast coils, but maximum local strain exceeds 2% somewhere during
  compression; the longerons fracture. Right regime, but longerons too thick
  or taper wrong.

**`sigma_crit`** — critical buckling stress in kPa. Compressive stress at
which the instability (coiling) initiates. Deterministic, directly computed
from FEM. `NaN` when `coilable == 0`. Designs with `coilable == 2` have a
`sigma_crit` value but are not mechanically usable.

**`energy`** — elastic energy absorption in kJ/m³. Area under the complete
compressive stress-strain curve. Stochastic (affected by manufacturing
imperfection simulation). `NaN` when `coilable == 0`.

---

## Objective

Find the design in the pool with the **highest `sigma_crit`** subject to
**`coilable == 1`**. Any design with `coilable != 1` is unusable for the
primary objective.

`energy` is a secondary objective: among coilable designs, higher energy
absorption is preferable, but only after maximising `sigma_crit`.

---

## Available data

All evaluations are pre-computed. There are no new simulations available.

The 1 000-sample dataset lives at:

```
experiment_data/experiment_data/
  input.csv    — 1 000 rows × 3 columns: ratio_d, ratio_pitch, ratio_top_diameter
  output.csv   — 1 000 rows × 3 columns: coilable, sigma_crit, energy
  domain.json  — f3dasm Domain definition (parameter bounds)
  jobs.csv     — job status metadata
```

---

## Evaluation budget and constraints

- **No new simulations.** The 1 000-sample dataset is the complete pool.
- Candidate designs must be evaluated by lookup against the pool. Proposals
  that do not match a row cannot be evaluated.
- The dataset was generated with a space-filling design; it covers the domain
  but is not dense in any particular region.

---

## Success criterion

**Primary:** Identify the single design with `coilable == 1` and the highest
`sigma_crit`. Report its exact parameter values and `sigma_crit`.

**Secondary:** If multiple designs have `sigma_crit` values near the optimum,
also report their `energy` values for trade-off analysis.

