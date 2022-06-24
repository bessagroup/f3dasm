import numpy as np
import pytest
from f3dasm.simulation.benchmark_functions import Levy
from f3dasm.src.simulation import Function

x = np.array([0.2, 0.3, 0.4, 0.6])  # 1D array with 4 dimensions
x1 = np.array([[0.1], [0.2], [0.3]])  # 2D array with 1 dimension
x2 = np.array([[0.0, 0.0], [0.5, 0.3], [1.0, 0.8]])  # 2D array with 2 dimensions
x3 = np.array(
    [[0.0, 0.0, 0.0], [0.5, 0.3, 0.2], [1.0, 0.8, 0.5]]
)  # 2D array with 3 dimensions


def test_f_not_implemented_error():
    class NewFunction(Function):
        pass

    with pytest.raises(NotImplementedError):
        f = NewFunction()
        y = f.eval(x)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (x, np.array([[0.62008502]])),
        (x1, np.array([[0.52179389], [0.42167184], [0.32794271]])),
        (x2, np.array([[0.71584455], [0.26789159], [0.00273873]])),
        (x3, np.array([[0.80668911], [0.37775304], [0.03987239]])),
    ],
)
def test_levy_noiseless(test_input, expected):
    f = Levy(noise=False)
    assert f.eval(test_input) == pytest.approx(expected)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (x, np.array([[0.6029378]])),
        (
            x1,
            np.array([[0.68073541], [0.40192466], [0.31258601]]),
        ),
        (
            x2,
            np.array([[0.9338950], [0.2553460], [0.00261048]]),
        ),
        (
            x3,
            np.array([[1.052411], [0.36006260], [0.038005265]]),
        ),
    ],
)
def test_levy_noise_20percent(test_input, expected):
    seed = 42
    f = Levy(noise=True, seed=seed)
    print(f.eval(test_input))
    assert f.eval(test_input) == pytest.approx(expected)


def test_input_vector_with_NaN_vales():
    f = Levy()
    x = np.array([1.0, 0.4, np.nan, 0.2])
    with pytest.raises(ValueError):
        y = f.eval(x)


def test_input_vector_with_inf_vales():
    f = Levy()
    x = np.array([1.0, 0.4, np.inf, 0.2])
    with pytest.raises(ValueError):
        y = f.eval(x)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
