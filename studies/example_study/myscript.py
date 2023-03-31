"""Example script to show the capabilities of multinode processing"""
# Standard
import logging
import time

import numpy as np
from pathos.helpers import mp

# Third-party
import f3dasm


def main_parallel(a: float, b: float, c: float) -> float:
    def evaluate_function(func, a, b, c):
        f = func(dimensionality=3, scale_bounds=np.tile([-1., 1.], (3, 1)))
        y = f(np.array([a, b, c])).ravel()[0]
        time.sleep(3)
        return y

    functions = [f3dasm.functions.Rastrigin, f3dasm.functions.Levy, f3dasm.functions.Ackley]
    with mp.Pool() as pool:
        y = pool.starmap(evaluate_function, [(func, a, b, c) for func in functions])

    # Sum the values
    out = sum(y)

    logging.info(f"Executed program with a={a:.3f}, b={b:.3f}, c={c:.3f}: \t Result: {out:.3f}")
    return out
