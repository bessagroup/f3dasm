import numpy as np
import pytest

from f3dasm.base.function import Function
from f3dasm.base.utils import read_pickle
from f3dasm.functions.pybenchfunction import Levy

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def script_loc(request):
    """Return the directory of the currently running test script"""

    # uses .join instead of .dirname so we get a LocalPath object instead of
    # a string. LocalPath.join calls normpath for us when joining the path
    return request.fspath.join("..")


@pytest.fixture
def levy_function():
    dim = 3
    seed = 42
    bounds = np.tile([-1.0, 1.0], (dim, 1))
    return Levy(dimensionality=dim, scale_bounds=bounds, seed=seed)


def test_loss_surface_levy_function(levy_function: Function, script_loc):
    _, _, Y = levy_function._create_mesh(px=300, domain=np.tile([-1.0, 1.0], (2, 1)))

    Y_check = read_pickle(script_loc.join("Levyfunction_2D"))
    assert (Y == Y_check).all()


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
