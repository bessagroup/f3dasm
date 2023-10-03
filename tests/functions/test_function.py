import numpy as np
import pytest

from f3dasm._src.datageneration.functions.pybenchfunction import Ackley

pytestmark = pytest.mark.smoke

def test_create_mesh_returns_correct_mesh_shape():
    # Arrange
    px = 10
    domain = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    instance = Ackley(dimensionality=2)

    # Act
    xv, yv, Y = instance._create_mesh(px, domain)

    # Assert
    assert xv.shape == (px, px)
    assert yv.shape == (px, px)
    assert Y.shape == (px, px)


def test_create_mesh_raises_value_error_with_invalid_px():
    # Arrange
    px = -10
    domain = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    instance = Ackley(dimensionality=2)

    # Act / Assert
    with pytest.raises(ValueError):
        instance._create_mesh(px, domain)


# def test_create_mesh_raises_value_error_with_invalid_domain_shape():
#     # Arrange
#     px = 10
#     domain = np.array([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]])
#     instance = Ackley(dimensionality=2)

#     # Act / Assert
#     with pytest.raises(ValueError):
#         instance._create_mesh(px, domain)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
