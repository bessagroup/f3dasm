# Black-Box Optimisation: 8-Dimensional Continuous Function

I have a real-valued function over an 8-dimensional continuous domain that I
want to minimise. I know it has multiple local minima but I do not know their
locations, values, or count. I have not solved this before.

**Goal: find the point x ∈ [−5, 5]⁸ that minimises f(x), and report its
coordinates and objective value.**

---

## Domain

Eight continuous inputs, each in [−5, 5]:

| Variable | Lower bound | Upper bound |
|----------|-------------|-------------|
| x1       | −5          | 5           |
| x2       | −5          | 5           |
| x3       | −5          | 5           |
| x4       | −5          | 5           |
| x5       | −5          | 5           |
| x6       | −5          | 5           |
| x7       | −5          | 5           |
| x8       | −5          | 5           |

---

## Evaluator

A pre-compiled evaluator lives in `workspace/`:

```python
import sys
sys.path.insert(0, "workspace")
from evaluator import evaluate   # evaluate(x: list[float]) -> float
```

`evaluate(x)` takes a list or array of exactly 8 floats and returns a scalar.
All inputs must lie within the bounds above. Behaviour outside [−5, 5]⁸ is
undefined.

The function is **deterministic** — identical inputs always return identical
outputs.

---

## Budget

**500 function evaluations.** The function is cheap to evaluate; the
constraint is the evaluation count, not wall time. Naive uniform sampling
across 8 dimensions will not reliably find the global minimum within this
budget. A strategy that adapts where it samples based on observed values will
do substantially better.

---

## What to deliver

Write your findings to `runs/<timestamp>/strategizer_notes/` as you go, and
report in `solution.md`:

- The best point (x1, …, x8) found and its f(x) value
- The total number of evaluations used
- The search strategy used and why
- Evidence the solution is not a local minimum (e.g. multiple restarts,
  deliberate exploration of distant regions)
