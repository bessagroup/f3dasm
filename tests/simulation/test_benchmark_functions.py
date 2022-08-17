import numpy as np
import pytest

pytestmark = pytest.mark.smoke

from f3dasm.functions.pybenchfunction import (
    Levy,
)
from f3dasm.base.function import Function

x = np.array([0.2, 0.3, 0.4, 0.6])  # 1D array with 4 dimensions
x1 = np.array([[0.1], [0.2], [0.3]])  # 2D array with 1 dimension
x2 = np.array([[0.0, 0.0], [0.5, 0.3], [1.0, 0.8]])  # 2D array with 2 dimensions
x3 = np.array([[0.0, 0.0, 0.0], [0.5, 0.3, 0.2], [1.0, 0.8, 0.5]])  # 2D array with 3 dimensions


# def test_f_not_implemented_error():
#     class NewFunction(Function):
#         pass

#     with pytest.raises(NotImplementedError):
#         f = NewFunction()
#         y = f.eval(x)


def test_input_vector_with_NaN_vales():
    f = Levy()
    x = np.array([1.0, 0.4, np.nan, 0.2])
    with pytest.raises(ValueError):
        y = f.__call__(x)


def test_input_vector_with_inf_vales():
    f = Levy()
    x = np.array([1.0, 0.4, np.inf, 0.2])
    with pytest.raises(ValueError):
        y = f.__call__(x)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
