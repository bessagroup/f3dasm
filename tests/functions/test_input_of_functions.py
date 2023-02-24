import numpy as np
import pytest

from f3dasm.functions.pybenchfunction import Levy

pytestmark = pytest.mark.smoke


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
