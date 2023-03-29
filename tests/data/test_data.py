import json

import pytest

from f3dasm.design.experimentdata import ExperimentData
from f3dasm.design.utils import create_experimentdata_from_json

pytestmark = pytest.mark.smoke


def test_get_output_data(data: ExperimentData):
    truth = data.data[[("output", "y1"), ("output", "y2")]]["output"]

    assert all(data.get_output_data() == truth)


def test_check_reproducibility(data: ExperimentData):
    data_json = data.to_json()
    data_new = create_experimentdata_from_json(data_json)
    assert all(data.data == data_new.data)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
