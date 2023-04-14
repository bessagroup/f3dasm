import json

import pytest

from f3dasm.design.experimentdata import ExperimentData

pytestmark = pytest.mark.smoke


def test_get_output_data(data: ExperimentData):
    truth = data.data[[("output", "y1"), ("output", "y2")]]["output"]

    assert all(data.get_output_data() == truth)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
