import numpy as np
import pytest

pytestmark = pytest.mark.smoke

from f3dasm.base.space import ContinuousParameter, DiscreteParameter
from f3dasm.base.design import DesignSpace
from f3dasm.base.data import Data
from f3dasm.base.function import Function
from f3dasm.sampling.samplers import RandomUniformSampling


@pytest.fixture
def data():
    seed = 42
    # Define the parameters
    x1 = ContinuousParameter(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = ContinuousParameter(name="x2", lower_bound=10.0, upper_bound=380.3)
    y = ContinuousParameter(name="y")

    # Create the design space
    input_space = [x1, x2]
    output_space = [y]
    design = DesignSpace(input_space=input_space, output_space=output_space)
    sampler = RandomUniformSampling(doe=design, seed=seed)
    data = sampler.get_samples(numsamples=20)

    return data


@pytest.fixture
def data_discrete():
    seed = 42
    # Define the parameters
    x1 = ContinuousParameter(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = ContinuousParameter(name="x2", lower_bound=10.0, upper_bound=380.3)
    x3 = DiscreteParameter(name="x3", lower_bound=10, upper_bound=380)
    y = ContinuousParameter(name="y")

    # Create the design space
    input_space = [x1, x2, x3]
    output_space = [y]
    design = DesignSpace(input_space=input_space, output_space=output_space)
    sampler = RandomUniformSampling(doe=design, seed=seed)
    data = sampler.get_samples(numsamples=20)

    return data


@pytest.fixture
def xsinx():
    class xsinx(Function):
        def f(self, x: np.ndarray) -> np.ndarray:

            n_points, n_features = np.shape(x)
            y = np.empty((n_points, 1))
            print(f"x-shape {np.shape(x)}, y-shape {np.shape(y)}")

            for ii in range(n_points):
                y[ii] = sum(x[ii, :] * np.sin(x[ii, :]))
            return y

    return xsinx


def test_data_to_numpy_array_benchmarkfunction(data: Data, xsinx: Function):
    func = xsinx()
    y = func.eval(input_x=data)


def test_data_to_numpy_array_benchmarkfunction_raise_error(data_discrete: Data, xsinx: Function):
    func = xsinx()
    with pytest.raises(TypeError):
        y = func.eval(input_x=data_discrete)


def test_data_and_numpy_input_eval(data: Data, xsinx: Function):
    x = data.get_input_data().to_numpy()
    func = xsinx()

    assert all(func.eval(x) == func.eval(data))


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
