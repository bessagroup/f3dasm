import numpy as np
from f3dasm.base.data import Data
from ..base.design import DesignSpace
from ..base.space import ContinuousParameter


def make_nd_continuous_design(bounds: np.ndarray, dimensions: int):
    input_space, output_space = [], []
    for dim in range(dimensions):
        input_space.append(ContinuousParameter(name=f"x{dim}", lower_bound=bounds[dim, 0], upper_bound=bounds[dim, 1]))

    output_space.append(ContinuousParameter(name="y"))

    return DesignSpace(input_space=input_space, output_space=output_space)


def _from_data_to_numpy_array_benchmarkfunction(
    data: Data,
) -> np.ndarray:
    """Check if doe is in right format"""
    if not data.designspace.is_single_objective_continuous():
        raise TypeError("All inputs and outputs need to be continuous parameters and output single objective")

    return data.get_input_data().to_numpy()


def _scale_vector(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Scale a vector x to a given scale"""
    return (scale[:, 1] - scale[:, 0]) * x + scale[:, 0]


def _descale_vector(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Inverse of the _scale_vector() function"""
    return (x - scale[:, 0]) / (scale[:, 1] - scale[:, 0])
