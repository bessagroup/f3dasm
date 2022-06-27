import numpy as np
import pytest

from f3dasm.src.space import ContinuousSpace, DiscreteSpace
from f3dasm.src.designofexperiments import DoE
from f3dasm.src.data import Data
from f3dasm.src.simulation import Function
from f3dasm.sampling.randomuniform import RandomUniform


@pytest.fixture
def data():
    seed = 42
    # Define the parameters
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = ContinuousSpace(name="x2", lower_bound=10.0, upper_bound=380.3)
    y = ContinuousSpace(name="y")

    # Create the design space
    input_space = [x1, x2]
    output_space = [y]
    design = DoE(input_space=input_space, output_space=output_space)
    sampler = RandomUniform(doe=design, seed=seed)
    samples = sampler.get_samples(numsamples=20)

    data = Data(doe=design)
    data.add(samples)

    return data


@pytest.fixture
def data_discrete():
    seed = 42
    # Define the parameters
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = ContinuousSpace(name="x2", lower_bound=10.0, upper_bound=380.3)
    x3 = DiscreteSpace(name="x3", lower_bound=10, upper_bound=380)
    y = ContinuousSpace(name="y")

    # Create the design space
    input_space = [x1, x2, x3]
    output_space = [y]
    design = DoE(input_space=input_space, output_space=output_space)
    sampler = RandomUniform(doe=design, seed=seed)
    samples = sampler.get_samples(numsamples=20)

    data = Data(doe=design)
    data.add(samples)

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
    y = func.eval(x=data)


def test_data_to_numpy_array_benchmarkfunction_raise_error(
    data_discrete: Data, xsinx: Function
):
    func = xsinx()
    with pytest.raises(TypeError):
        y = func.eval(x=data_discrete)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
