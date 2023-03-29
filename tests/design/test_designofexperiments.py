import json

import pandas as pd
import pytest

from f3dasm.design.utils import create_design_from_json
from f3dasm.design.design import DesignSpace
from f3dasm.design.parameter import (CategoricalParameter, ContinuousParameter,
                                     DiscreteParameter)

pytestmark = pytest.mark.smoke


def test_empty_space_doe():
    doe = DesignSpace()
    empty_list = []
    assert doe.input_space == empty_list


def test_correct_doe(doe):
    pass


def test_get_continuous_parameters(doe: DesignSpace):
    x1 = ContinuousParameter(name="x1", lower_bound=2.4, upper_bound=10.3)
    x3 = ContinuousParameter(name="x3", lower_bound=10.0, upper_bound=380.3)
    assert doe.get_continuous_input_parameters() == [x1, x3]


def test_get_discrete_parameters(doe: DesignSpace):
    x2 = DiscreteParameter(name="x2", lower_bound=5, upper_bound=80)
    x5 = DiscreteParameter(name="x5", lower_bound=2, upper_bound=3)
    assert doe.get_discrete_input_parameters() == [x2, x5]


def test_get_categorical_parameters(doe: DesignSpace):
    x4 = CategoricalParameter(name="x4", categories=["test1", "test2", "test3"])
    assert doe.get_categorical_input_parameters() == [x4]


def test_get_continuous_names(doe: DesignSpace):
    assert doe.get_continuous_input_names() == ["x1", "x3"]


def test_get_discrete_names(doe: DesignSpace):
    assert doe.get_discrete_input_names() == ["x2", "x5"]


def test_get_categorical_names(doe: DesignSpace):
    assert doe.get_categorical_input_names() == ["x4"]


def test_add_input_space():
    x1 = ContinuousParameter(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteParameter(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousParameter(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalParameter(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteParameter(name="x5", lower_bound=2, upper_bound=3)
    designspace = [x1, x2, x3]

    design = DesignSpace(input_space=designspace)
    design.add_input_space(x4)
    design.add_input_space(x5)

    assert design.input_space == [x1, x2, x3, x4, x5]


def test_add_output_space():
    x1 = ContinuousParameter(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteParameter(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousParameter(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalParameter(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteParameter(name="x5", lower_bound=2, upper_bound=3)
    designspace = [x1, x2, x3]

    design = DesignSpace(output_space=designspace)
    design.add_output_space(x4)
    design.add_output_space(x5)

    assert design.output_space == [x1, x2, x3, x4, x5]


def test_getNumberOfInputParameters(doe: DesignSpace):
    assert doe.get_number_of_input_parameters() == 5


def test_getNumberOfOutputParameters(doe: DesignSpace):
    assert doe.get_number_of_output_parameters() == 2


def test_get_input_space(doe: DesignSpace):
    assert doe.input_space == doe.get_input_space()


def test_get_output_space(doe: DesignSpace):
    assert doe.output_space == doe.get_output_space()


def test_all_input_continuous_False(doe: DesignSpace):
    assert doe._all_input_continuous() is False


def test_all_input_continuous_True():
    x1 = ContinuousParameter(name="x1", lower_bound=2.4, upper_bound=10.3)
    x3 = ContinuousParameter(name="x3", lower_bound=10.0, upper_bound=380.3)

    y1 = ContinuousParameter(name="y1")
    y2 = ContinuousParameter(name="y2")
    designspace = [x1, x3]
    output_space = [y1, y2]

    doe = DesignSpace(input_space=designspace, output_space=output_space)

    assert doe._all_input_continuous() is True


def test_cast_types_dataframe_input(doe: DesignSpace):
    ground_truth = {
        ("output", "y1"): "float",
        ("output", "y2"): "float",
    }

    assert doe._cast_types_dataframe(space=doe.output_space, label="output") == ground_truth


def test_cast_types_dataframe_output(doe: DesignSpace):
    ground_truth = {
        ("input", "x1"): "float",
        ("input", "x2"): "int",
        ("input", "x3"): "float",
        ("input", "x4"): "category",
        ("input", "x5"): "int",
    }

    assert doe._cast_types_dataframe(space=doe.input_space, label="input") == ground_truth


def test_same_name_of_parameters_error():
    x1 = ContinuousParameter(name="x0", lower_bound=2.4, upper_bound=10.3)
    x2 = ContinuousParameter(name="x0", lower_bound=10.0, upper_bound=380.3)

    y1 = ContinuousParameter(name="y1")
    designspace = [x1, x2]
    output_space = [y1]
    with pytest.raises(ValueError):
        doe = DesignSpace(input_space=designspace, output_space=output_space)


def test_check_reproducibility(doe: DesignSpace):
    design_json = doe.to_json()
    doe_new = create_design_from_json(design_json)
    assert doe == doe_new


def test_to_json(design_space):
    # Ensure that to_json method returns valid JSON
    json.loads(design_space.to_json())


def test_get_input_space_2(design_space):
    # Ensure that get_input_space returns the input space
    assert design_space.get_input_space() == design_space.input_space


def test_get_output_space_2(design_space):
    # Ensure that get_output_space returns the output space
    assert design_space.get_output_space() == design_space.output_space


def test_get_output_names(design_space):
    # Ensure that get_output_names returns the correct output parameter names
    assert design_space.get_output_names() == [("output", "y")]


def test_get_input_names(design_space):
    # Ensure that get_input_names returns the correct input parameter names
    assert design_space.get_input_names() == [("input", "x1"), ("input", "x2"), ("input", "x3")]


def test_is_single_objective_continuous(design_space):
    # Ensure that is_single_objective_continuous correctly identifies single objective continuous output spaces
    assert not design_space.is_single_objective_continuous()


def test_get_number_of_input_parameters(design_space):
    # Ensure that get_number_of_input_parameters returns the correct number of input parameters
    assert design_space.get_number_of_input_parameters() == 3


def test_get_number_of_output_parameters(design_space):
    # Ensure that get_number_of_output_parameters returns the correct number of output parameters
    assert design_space.get_number_of_output_parameters() == 1


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
