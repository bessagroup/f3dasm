# Modular Resonance Search

I am studying a small two-parameter integer optimisation problem. The problem is inspired by the structure of Project Euler #952 ("Order Modulo Factorial") but the question I want answered is different and the search space is small enough to explore with sampling. **I have not solved it.** I do not know what the optimum looks like and I have no candidate (k, m) pair I am protecting.

## The function

For positive integers `k` and `m`, define the multiplicative order

```
ord(k, m)  =  the smallest positive integer r such that k^r ≡ 1  (mod m)
```

when `gcd(k, m) = 1`. When `gcd(k, m) > 1` the multiplicative order is undefined; treat such `(k, m)` pairs as **infeasible** (assign objective `−∞` or any large negative sentinel).

The objective I want to maximise is

```
resonance(k, m)  =  ord(k, m) / ln(m)
```

over the search domain

```
k  ∈  {2, 3, …, 50}            (integer)
m  ∈  {1000, 1001, …, 100000}  (integer)
```

So the feasible domain has ≈ 49 × 99 001 ≈ 4.85 million points. I cannot afford to brute-force all of them on a single machine — I need a sampled / optimised search.

## What I want from you

A defensible answer of the form **"the largest resonance(k, m) you observed during the search, and the (k, m) at which it was observed"**, together with the search strategy you used and any evidence about whether the value you reported is likely to be the global optimum or just a local one. I will read your solution.md as the primary artefact.

## Resources and constraints

- This problem is closed-form once `ord(k, m)` is implementable, but `ord(k, m)` for `m` up to 10^5 requires care — a naive `r = 1; while pow(k, r, m) != 1: r += 1` is correct but can be slow for cases where `r` is large. Faster: factor `λ(m)` (Carmichael) and test divisors of `λ(m)` in ascending order. Use whichever you can implement cleanly.
- **You must use `f3dasm` for the search.** Specifically:
  - Build a `Domain` with two `add_int` parameters (`k` and `m`) and one `add_output("resonance")`.
  - Wrap `ord(k, m)` / `resonance(k, m)` in a `DataGenerator` subclass (its `execute` method takes an `ExperimentSample`, computes the score, sets `_output_data["resonance"]`).
  - Sample candidate `(k, m)` points using one of `f3dasm`'s samplers (`Latin`, `Sobol`, `RandomUniform`, `Grid`) — your choice — and evaluate them through your `DataGenerator`. Iterate / refine as needed.
  - Use `ExperimentData` to store every evaluation and to compute the running best.
- Everything you write — scripts, intermediate data, plots, logs — must live under `workspace/`. **Files written anywhere else (for example `/tmp/`) will not be included in the deliverable and the run will be unreproducible.**
- The replication script the user can run after the fact must live at `workspace/replicate.py` and, when executed (`python workspace/replicate.py`), must re-derive the same best `(k, m, resonance)` triple you report in `solution.md` (or, if you used random sampling without seeding, a value within a documented tolerance).
- You may install no Python packages. `f3dasm`, `numpy`, and the standard library are enough.

## What I am unsure about (you might ask me)

- Whether the secondary objective (e.g. the *second*-largest resonance value with a constraint that `k` differs by ≥ 3) is interesting to me. Default: no, only the primary maximum.
- Whether to bound the search budget by total evaluations or by wall time. Default: ≤ 5000 evaluations, ≤ 10 minutes — whichever comes first.
- Whether co-primality should be enforced before evaluation. Default: do whatever's simpler; infeasible points just get the sentinel.
