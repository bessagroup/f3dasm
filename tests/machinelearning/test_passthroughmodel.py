import numpy as np
import pytest

from f3dasm.base.evaluator import Evaluator
from f3dasm.base.utils import convert_autograd_to_tensorflow
from f3dasm.design import DesignSpace
from f3dasm.functions import Ackley, Function, Levy, Rosenbrock
from f3dasm.optimization import OPTIMIZERS, Optimizer
from f3dasm.sampling import LatinHypercube

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("pybenchfunction", [Levy, Ackley, Rosenbrock])
def test_passthroughmodel(pybenchfunction: Function, design_2d: DesignSpace):
    x = np.array([[0.0, 0.1]])
    loss_function = pybenchfunction(dimensionality=design_2d.get_number_of_input_parameters(),
                                    scale_bounds=design_2d.get_bounds())
    f = convert_autograd_to_tensorflow(loss_function.__call__)
    evaluator = Evaluator(loss_function=f)
    loss_new, _ = evaluator.evaluate(x)

    loss_old = loss_function(x)
    assert loss_new == loss_old


@pytest.mark.parametrize("pybenchfunction", [Levy, Ackley, Rosenbrock])
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_passthrough_optimizer(pybenchfunction: Function, optimizer, design_2d: DesignSpace):
    x = np.array([[0.0, 0.1]])
    loss_function = pybenchfunction(dimensionality=design_2d.get_number_of_input_parameters(),
                                    scale_bounds=design_2d.get_bounds())
    f = convert_autograd_to_tensorflow(loss_function.__call__)
    evaluator = Evaluator(loss_function=f)
    data = LatinHypercube(design=design_2d, seed=42).get_samples(30)
    w = data.get_n_best_input_parameters_numpy(30)
    data.add_output(output=evaluator(w))

    opt: Optimizer = optimizer(data=data, seed=42)
    opt.iterate(iterations=10, function=evaluator)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
