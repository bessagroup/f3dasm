import numpy as np
import pytest

from f3dasm import make_nd_continuous_design
from f3dasm.machinelearning.evaluator import Evaluator
from f3dasm.data.linearregression_data import LinearRegressionData
from f3dasm.machinelearning.linear_regression import LinearRegression
from f3dasm.optimization import OPTIMIZERS, Optimizer
from f3dasm.sampling import LatinHypercube

pytestmark = pytest.mark.smoke


def test_linearregression():
    DIMENSIONALITY = 5

    random_weights = list(np.random.random(size=DIMENSIONALITY+1))
    n = 100
    b = random_weights[0]
    w = random_weights[1:]
    learning_data = LinearRegressionData(n=n, b=b, w=w)

    model = LinearRegression(dimensionality=DIMENSIONALITY)
    evaluator = Evaluator(learning_data=learning_data, model=model)
    x = np.array([0.2]*(DIMENSIONALITY+1))
    y, dydx = evaluator.evaluate(x)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_linearregression_optimizer(optimizer: Optimizer):
    DIMENSIONALITY = 5

    random_weights = list(np.random.random(size=DIMENSIONALITY+1))
    n = 100
    b = random_weights[0]
    w = random_weights[1:]
    learning_data = LinearRegressionData(n=n, b=b, w=w)

    model = LinearRegression(dimensionality=DIMENSIONALITY)
    evaluator = Evaluator(learning_data=learning_data, model=model)
    x = np.array([0.2]*(DIMENSIONALITY+1))
    y, dydx = evaluator.evaluate(x)

    design = make_nd_continuous_design(dimensionality=DIMENSIONALITY+1, bounds=np.tile([-1.0, 1.0],
                                                                                       (DIMENSIONALITY+1, 1)))
    data = LatinHypercube(design=design, seed=42).get_samples(30)
    w = data.get_n_best_input_parameters_numpy(30)
    data.add_output(output=evaluator(w))

    opt = optimizer(data=data, seed=42)
    opt.iterate(iterations=10, function=evaluator)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
