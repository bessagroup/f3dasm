"""One-time setup: generate and compile the black-box evaluator.

Run once from the study root before starting the agentic run:

    python generate_evaluator.py

Produces workspace/evaluator.so (Linux) or workspace/evaluator.dylib
(macOS) and workspace/evaluator.py.  The C source is written, compiled,
and immediately deleted — the binary is the only persistent record of
the function.
"""

#                                                                       Modules
# =============================================================================

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

#                                                               Study constants
# =============================================================================

_SEED = 20260519
_DIM = 8
_N_BUMPS = 25
_SIGMA_BOUNDS = (0.3, 1.2)
_WEIGHT_BOUNDS = (0.5, 2.0)
_CENTER_BOUNDS = (-4.0, 4.0)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


# -----------------------------------------------------------------------------
# Parameter generation
# -----------------------------------------------------------------------------

def _generate_params():
    rng = np.random.default_rng(_SEED)
    centers = rng.uniform(*_CENTER_BOUNDS, (_N_BUMPS, _DIM))
    weights = rng.uniform(*_WEIGHT_BOUNDS, _N_BUMPS)
    sigmas = rng.uniform(*_SIGMA_BOUNDS, _N_BUMPS)
    return centers, weights, sigmas


# -----------------------------------------------------------------------------
# C source construction
# -----------------------------------------------------------------------------

def _c_literal(v: float) -> str:
    return f"{v:.17g}"


def _c_array_1d(name: str, values) -> str:
    vals = ", ".join(_c_literal(v) for v in values)
    return f"static const double {name}[{len(values)}] = {{{vals}}};"


def _c_array_2d(name: str, rows) -> str:
    n, m = len(rows), len(rows[0])
    inner = ",\n    ".join(
        "{" + ", ".join(_c_literal(v) for v in row) + "}"
        for row in rows
    )
    return f"static const double {name}[{n}][{m}] = {{\n    {inner}\n}};"


def _build_c_source(centers, weights, sigmas) -> str:
    return textwrap.dedent(f"""\
        #include <math.h>
        #include <stddef.h>

        #define N_BUMPS {_N_BUMPS}
        #define DIM     {_DIM}

        {_c_array_2d("centers", centers)}
        {_c_array_1d("weights", weights)}
        {_c_array_1d("sigmas",  sigmas)}

        double evaluate(const double *x) {{
            double result = 0.0;
            int i, d;
            for (i = 0; i < N_BUMPS; i++) {{
                double sq_dist = 0.0;
                for (d = 0; d < DIM; d++) {{
                    double diff = x[d] - centers[i][d];
                    sq_dist += diff * diff;
                }}
                result -= weights[i]
                          * exp(-sq_dist
                                / (2.0 * sigmas[i] * sigmas[i]));
            }}
            return result;
        }}
    """)


# -----------------------------------------------------------------------------
# Python wrapper generation
# -----------------------------------------------------------------------------

def _write_wrapper(lib_name: str, workspace: Path) -> None:
    src = textwrap.dedent(f"""\
        \"\"\"ctypes wrapper for the compiled black-box evaluator.\"\"\"

        import ctypes
        from pathlib import Path

        _here = Path(__file__).parent
        _candidates = ("evaluator.dylib", "evaluator.so")
        _lib = None
        for _name in _candidates:
            _p = _here / _name
            if _p.exists():
                _lib = ctypes.CDLL(str(_p))
                break
        if _lib is None:
            raise FileNotFoundError(
                f"No compiled evaluator found in {{_here}}. "
                "Run generate_evaluator.py first."
            )

        _lib.evaluate.restype = ctypes.c_double
        _lib.evaluate.argtypes = [ctypes.POINTER(ctypes.c_double)]


        def evaluate(x):
            \"\"\"Evaluate the black-box function at x.

            Parameters
            ----------
            x : sequence of {_DIM} float
                Point in [-5, 5]^{_DIM}.

            Returns
            -------
            float
                Objective value (minimise).
            \"\"\"
            arr = (ctypes.c_double * {_DIM})(*x)
            return float(_lib.evaluate(arr))
    """)
    (workspace / "evaluator.py").write_text(src)


# -----------------------------------------------------------------------------
# Compilation
# -----------------------------------------------------------------------------

def _compile(c_path: Path, workspace: Path) -> Path:
    if sys.platform == "darwin":
        lib_name = "evaluator.dylib"
        cmd = [
            "gcc", "-O2", "-dynamiclib",
            "-o", str(workspace / lib_name),
            str(c_path), "-lm",
        ]
    else:
        lib_name = "evaluator.so"
        cmd = [
            "gcc", "-O2", "-shared", "-fPIC",
            "-o", str(workspace / lib_name),
            str(c_path), "-lm",
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    c_path.unlink()

    if result.returncode != 0:
        raise RuntimeError(
            f"Compilation failed.\n\nstderr:\n{result.stderr}"
        )

    return workspace / lib_name


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    study_root = Path(__file__).parent
    workspace = study_root / "workspace"
    workspace.mkdir(exist_ok=True)

    centers, weights, sigmas = _generate_params()
    c_source = _build_c_source(centers, weights, sigmas)

    c_path = workspace / "_evaluator_src.c"
    c_path.write_text(c_source)

    lib_path = _compile(c_path, workspace)
    _write_wrapper(lib_path.name, workspace)

    print(f"Compiled  : {lib_path}")
    print(f"Wrapper   : {workspace / 'evaluator.py'}")
    print()
    print("Study is ready. Start the agentic run with:")
    print(f"  python -m f3dasm.agentic {study_root}")


if __name__ == "__main__":
    main()
