# Problem Briefing: Supercompressible Metamaterial Design (7D)

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
critical buckling stress before instability onset.

The longeron cross-section shape is unrestricted: area, two area moments of
inertia, and polar moment are all free parameters, as is the material's shear
modulus ratio. `ratio_Ixx` and `ratio_Iyy` are independent — the cross-section
is not constrained to be symmetric.

---

## Design space

Seven continuous input parameters define the geometry and material of the
longerons. All other parameters are fixed and must never be proposed.

| Parameter              | Bounds                         | Physical meaning                                                                        |
|------------------------|--------------------------------|-----------------------------------------------------------------------------------------|
| `ratio_area`           | [1.17×10⁻⁵, 4.1×10⁻³]         | Cross-sectional area of a longeron relative to the square of the bottom ring diameter   |
| `ratio_Ixx`            | [1.128×10⁻¹¹, 1.4×10⁻⁶]       | Second moment of area about the x-axis, relative to D1⁴                                |
| `ratio_Iyy`            | [1.128×10⁻¹¹, 1.4×10⁻⁶]       | Second moment of area about the y-axis, relative to D1⁴ (independent of Ixx)           |
| `ratio_J`              | [1.353×10⁻¹¹, 7.77×10⁻⁶]      | Polar moment of area, relative to D1⁴                                                  |
| `ratio_pitch`          | [0.25, 1.50]                   | Mast height relative to bottom diameter: P / D1                                         |
| `ratio_top_diameter`   | [0.0, 0.8]                     | Taper of the mast: (D1 − D2) / D1                                                      |
| `ratio_shear_modulus`  | [0.035, 0.45]                  | Shear modulus relative to the Young's modulus: G / E                                   |

Fixed constants (do not vary these):

| Constant           | Value    |
|--------------------|----------|
| `young_modulus`    | 3500 MPa |
| `n_longerons`      | 3        |
| `bottom_diameter`  | 100 mm   |

Note: `circular` is `False` — the cross-section shape is fully determined by
the four ratio parameters above and is not constrained to a circle.

---

## Outputs and what they mean

Each evaluated design produces three output values:

**`coilable`** — integer with three possible values:
- `0` — the mast buckles in the wrong mode; it does not coil. Fundamental
  geometry failure.
- `1` — the mast coils reversibly; maximum local strain during full compression
  stays below 2%, so PLA does not fracture. The only acceptable outcome.
- `2` — the mast coils, but maximum local strain exceeds 2%; longerons
  fracture. Right regime, but cross-section too large or taper wrong.

**`sigma_crit`** — critical buckling stress in kPa. Compressive stress at
which coiling initiates. Deterministic, from FEM. `NaN` when `coilable == 0`.
Designs with `coilable == 2` have a value but are not mechanically usable.

**`energy`** — elastic energy absorption in kJ/m³. Area under the compressive
stress-strain curve. Stochastic. `NaN` when `coilable == 0`.

---

## Objective

Find the design in the pool with the **highest `sigma_crit`** subject to
**`coilable == 1`**. Any design with `coilable != 1` is unusable.

`energy` is a secondary objective: among coilable designs, higher energy
absorption is preferable, but only after maximising `sigma_crit`.

---

## Available data

All evaluations are pre-computed. There are no new simulations available.

The 50 000-sample dataset lives at:

```
experiment_data/experiment_data/
  input.csv    — 50 000 rows × 7 columns: ratio_area, ratio_Ixx, ratio_Iyy,
                 ratio_J, ratio_pitch, ratio_top_diameter, ratio_shear_modulus
  output.csv   — 50 000 rows × 3 columns: coilable, sigma_crit, energy
  domain.json  — f3dasm Domain definition (parameter bounds)
  jobs.csv     — job status metadata
```

---

## Evaluation budget and constraints

- **No new simulations.** The 50 000-sample dataset is the complete pool.
- Candidate designs must be evaluated by lookup against the pool.
- The dataset was generated with a space-filling design across all 7 dimensions.

---

## Success criterion

**Primary:** Identify the single design with `coilable == 1` and the highest
`sigma_crit`. Report all seven parameter values and `sigma_crit`.

**Secondary:** If multiple designs have `sigma_crit` values near the optimum,
also report their `energy` values for trade-off analysis.

