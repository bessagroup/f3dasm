import numpy as np
import pytest

from f3dasm.base.evaluator import Evaluator
from f3dasm.base.utils import convert_autograd_to_tensorflow
from f3dasm.design import DesignSpace
from f3dasm.functions import Ackley, Levy, Rosenbrock

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("pybenchfunction", [Levy, Ackley, Rosenbrock])
def test_passthroughmodel(pybenchfunction, design_2d: DesignSpace):
    x = np.array([[0.0, 0.1]])
    loss_function = pybenchfunction(dimensionality=design_2d.get_number_of_input_parameters(),
                                    scale_bounds=design_2d.get_bounds())
    f = convert_autograd_to_tensorflow(loss_function.__call__)
    evaluator = Evaluator(loss_function=f)
    loss_new, _ = evaluator.evaluate(x)

    loss_old = loss_function(x)
    assert loss_new == loss_old


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
