import pytest

from f3dasm.machinelearning import Model
from f3dasm.machinelearning.all_models import MODELS

pytestmark = pytest.mark.smoke


def test_models_exist():
    assert MODELS, "No models were found"


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
