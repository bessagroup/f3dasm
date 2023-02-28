import numpy as np
import pytest

from f3dasm.base.evaluator import Evaluator
from f3dasm.data.linearregression_data import LinearRegressionData
from f3dasm.machinelearning.linear_regression import LinearRegression

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


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
