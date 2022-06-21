import numpy as np
import pytest
from f3dasm.simulation.benchmark_functions import Levy

x = np.array([0.2, 0.3, 0.4, 0.6])  # 1D array with 4 dimensions
x1 = np.array([[0.1], [0.2], [0.3]])  # 2D array with 1 dimension
x2 = np.array([[0.0, 0.0], [0.5, 0.3], [1.0, 0.8]])  # 2D array with 2 dimensions
x3 = np.array(
    [[0.0, 0.0, 0.0], [0.5, 0.3, 0.2], [1.0, 0.8, 0.5]]
)  # 2D array with 3 dimensions


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (x, np.array([[0.62008502]])),
        (x1, np.array([[0.52179389], [0.42167184], [0.32794271]])),
        (x2, np.array([[0.71584455], [0.26789159], [0.00273873]])),
        (x3, np.array([[0.80668911], [0.37775304], [0.03987239]])),
    ],
)
def test_levy(test_input, expected):
    f = Levy()
    assert f.eval(test_input) == pytest.approx(expected)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
