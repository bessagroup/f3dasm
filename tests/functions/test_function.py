import numpy as np
import pytest

from f3dasm import ExperimentData
from f3dasm._src.datageneration.functions.pybenchfunction import Ackley
from f3dasm.design import make_nd_continuous_domain

pytestmark = pytest.mark.smoke


def test_create_mesh_returns_correct_mesh_shape():
    # Arrange
    px = 10
    arr = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    domain = make_nd_continuous_domain(bounds=arr)
    instance = Ackley()
    instance.init(data=ExperimentData(domain=domain))
    # Act
    xv, yv, Y = instance._create_mesh(px, arr)

    # Assert
    assert xv.shape == (px, px)
    assert yv.shape == (px, px)
    assert Y.shape == (px, px)


def test_create_mesh_raises_value_error_with_invalid_px():
    # Arrange
    px = -10
    arr = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    domain = make_nd_continuous_domain(bounds=arr)
    instance = Ackley()

    instance.init(data=ExperimentData(domain=domain))

    # Act / Assert
    with pytest.raises(ValueError):
        instance._create_mesh(px, arr)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
