# Problem Briefing: Supercompressible Metamaterial Design

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

Three continuous input parameters define the geometry. All other geometric
and material parameters are fixed and must never be proposed.

| Parameter           | Bounds       | Physical meaning                                        |
|---------------------|--------------|--------------------------------------------------------|
| `ratio_d`           | [0.004, 0.073] | Longeron cross-section diameter relative to bottom ring diameter: d / D1 |
| `ratio_pitch`       | [0.25, 1.50]   | Mast height relative to bottom diameter: P / D1       |
| `ratio_top_diameter`| [0.0, 0.8]     | Taper of the mast: (D1 − D2) / D1, where D1 is the bottom ring diameter and D2 is the top ring diameter |

Fixed constants (do not vary these):
- `young_modulus = 3500.0` MPa
- `n_longerons = 3`
- `bottom_diameter = 100.0` mm
- `ratio_shear_modulus = 0.3677`
- `circular = True` (longerons have circular cross-sections)

---

## Outputs and what they mean

Each evaluated design produces three output values:

**`coilable`** — integer with three possible values:
- `0` — the mast buckles in the wrong mode (global bending, local longeron
  buckling); it does not coil at all. This is a fundamental geometry failure.
- `1` — the mast coils reversibly when compressed; maximum local strain
  throughout full compression stays below 2%, so PLA does not fracture. This
  is the only acceptable outcome.
- `2` — the mast coils, but maximum local strain exceeds 2% somewhere during
  compression; the longerons fracture. The geometry is in the right coiling
  regime but the longerons are too thick or the taper is wrong.

**`sigma_crit`** — critical buckling stress in kPa. This is the compressive
stress at which the instability (coiling) initiates. It is deterministic and
directly computed from the FEM analysis. It is `NaN` whenever `coilable == 0`
because designs that do not coil do not produce a meaningful buckling stress in
this context. Designs with `coilable == 2` do have a `sigma_crit` value, but
they are not mechanically usable.

**`energy`** — elastic energy absorption in kJ/m³. This is the area under the
complete compressive stress-strain curve. It is stochastic (affected by
manufacturing imperfections in the simulation) and is `NaN` when `coilable == 0`.

---

## Objective

Find the design in the pool with the **highest `sigma_crit`** subject to
**`coilable == 1`**. Any design with `coilable != 1` is unusable for the
primary objective — I leave it to you to decide how to handle infeasible
designs in your search strategy.

The `energy` output is a secondary objective: among coilable designs, higher
energy absorption is preferable, but I care more about `sigma_crit`. Do not
optimize for `energy` at the expense of `sigma_crit`.

---

## Available data

All evaluations are pre-computed. There are no new simulations available.
The dataset is the entire evaluatable universe of designs for this problem.

The 1000-sample 3D dataset lives here:

```
studies/fragile_becomes_supercompressible/experiment_data/supercompressible_3d/experiment_data/
  input.csv    — 1000 rows, columns: ratio_d, ratio_pitch, ratio_top_diameter
  output.csv   — 1000 rows, columns: coilable, sigma_crit, energy
  domain.json  — f3dasm Domain definition (parameter bounds)
  jobs.csv     — job status metadata (likely not needed for analysis)
```

There is also a paper summary at:
```
studies/fragile_becomes_supercompressible/bessa2019.md
```
This paper describes the original study that introduced this metamaterial
concept and used a data-driven approach to design it. It is provided as
scientific background only — **not as a target or anchor**. Do not treat any
design mentioned in that paper as a goal. I want you to find the best design
in the pool independently.

---

## Evaluation budget and constraints

- There are **no new simulations**. You cannot query Abaqus or run new FEM
  evaluations. The 1000-sample dataset is the complete pool.
- Candidate designs you propose must be evaluated by lookup against this pool.
  Proposals that do not match a row in the pool cannot be evaluated.
- The dataset was generated with a space-filling design; it covers the domain
  but is not dense in any particular region.

---

## Success criterion

**Primary:** Identify the single design in the 1000-sample pool that has
`coilable == 1` and the highest `sigma_crit`. Report its exact parameter
values (`ratio_d`, `ratio_pitch`, `ratio_top_diameter`) and its `sigma_crit`
value.

**Secondary:** If multiple designs have `sigma_crit` values near the optimum,
also report their `energy` values so I can consider the trade-off.

---

## What I know and do not know

I have **not solved this before**. I do not know which region of the design
space produces `coilable == 1` designs, or where in that region `sigma_crit`
is highest. I have an intuition that the feasible region (coilable == 1) may
be non-trivially shaped — some combinations of parameters that look
geometrically reasonable likely fail in unexpected ways. But I have no prior
solution to share.

I can answer questions about the dataset structure, the file paths, and the
column definitions. I **cannot** answer detailed questions about the FEM
model, the physical mechanics of buckling, or the manufacturing process. If
you need that background, the paper summary at `bessa2019.md` may help, but
treat it as context rather than guidance.
