import numpy as np

from f3dasm.data import LinearRegressionData


def test_LinearRegressionData_create():
    # Test that data is created correctly
    n = 100
    b = 2.0
    w = [1.0, 2.0, 3.0]
    noise_multiplier = 0.01
    data = LinearRegressionData(n=n, b=b, w=w, noise_multiplier=noise_multiplier)

    assert data.X.shape == (n, len(w))
    assert data.y.shape == (n, 1)

    expected_y = np.matmul(data.X, np.reshape(w, (-1, 1))) + b
    noise = (np.random.normal(size=(n, 1)) * noise_multiplier)
    assert np.allclose(data.y, expected_y + noise, atol=.1)


def test_LinearRegressionData_get_input_data():
    # Test that get_input_data() returns X
    n = 100
    b = 2.0
    w = [1.0, 2.0, 3.0]
    noise_multiplier = 0.01
    data = LinearRegressionData(n=n, b=b, w=w, noise_multiplier=noise_multiplier)

    assert np.allclose(data.get_input_data(), data.X, atol=1e-2)


def test_LinearRegressionData_get_labels():
    # Test that get_labels() returns y
    n = 100
    b = 2.0
    w = [1.0, 2.0, 3.0]
    noise_multiplier = 0.01
    data = LinearRegressionData(n=n, b=b, w=w, noise_multiplier=noise_multiplier)

    assert np.allclose(data.get_labels(), data.y, atol=1e-2)
