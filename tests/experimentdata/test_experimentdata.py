# from __future__ import annotations

# import csv
# import pickle
# from pathlib import Path
# from typing import Callable, Iterable, Union

# import numpy as np
# import pandas as pd
# import pytest
# import xarray as xr

# from f3dasm import ExperimentData, ExperimentSample
# from f3dasm._src.design.parameter import ContinuousParameter
# from f3dasm._src.experimentdata.utils import DataTypes
# from f3dasm.design import Domain, make_nd_continuous_domain

# pytestmark = pytest.mark.smoke

# SEED = 42


# def test_check_experimentdata(experimentdata: ExperimentData):
#     assert isinstance(experimentdata, ExperimentData)

# # Write test functions


# def test_experiment_data_init(experimentdata: ExperimentData, domain: Domain):
#     assert experimentdata.domain == domain
#     assert experimentdata.project_dir == Path.cwd()
#     # Add more assertions as needed


# def test_experiment_data_add(experimentdata: ExperimentData,
#                              experimentdata2: ExperimentData, domain: Domain):
#     experimentdata_total = ExperimentData(domain)
#     experimentdata_total.add_experiments(experimentdata)
#     experimentdata_total.add_experiments(experimentdata2)
#     assert experimentdata_total == experimentdata + experimentdata2


# def test_experiment_data_len_empty(domain: Domain):
#     experiment_data = ExperimentData(domain)
#     assert len(experiment_data) == 0  # Update with the expected length


# def test_experiment_data_len_equals_input_data(experimentdata: ExperimentData):
#     assert len(experimentdata) == len(experimentdata.data)


# @pytest.mark.parametrize("slice_type", [3, [0, 1, 3]])
# def test_experiment_data_select(slice_type: int | Iterable[int], experimentdata: ExperimentData):
#     sliced_experimentdata = experimentdata.select(slice_type)
#     constructed_experimentdata = ExperimentData.from_data(data=sliced_experimentdata.data,
#                                                           domain=experimentdata.domain)
#     assert sliced_experimentdata == constructed_experimentdata

# #                                                                           Constructors
# # ======================================================================================


# def test_from_file(experimentdata_continuous: ExperimentData, seed: int, tmp_path: Path):
#     # experimentdata_continuous.filename = tmp_path / 'test001'
#     experimentdata_continuous.store(tmp_path / 'experimentdata')

#     experimentdata_from_file = ExperimentData.from_file(
#         tmp_path / 'experimentdata')

#     experimentdata_from_file.round(3)

#     assert experimentdata_from_file == experimentdata_continuous
#     # # Check if the input_data attribute of ExperimentData matches the expected_data
#     # pd.testing.assert_frame_equal(
#     #     experimentdata_continuous._input_data.to_dataframe(), experimentdata_from_file._input_data.to_dataframe(), check_dtype=False, atol=1e-6)
#     # pd.testing.assert_frame_equal(experimentdata_continuous._output_data.to_dataframe(),
#     #                               experimentdata_from_file._output_data.to_dataframe())
#     # pd.testing.assert_series_equal(
#     #     experimentdata_continuous._jobs.jobs, experimentdata_from_file._jobs.jobs)
#     # # assert experimentdata_continuous.input_data == experimentdata_from_file.input_data
#     # assert experimentdata_continuous._output_data == experimentdata_from_file._output_data
#     # assert experimentdata_continuous.domain == experimentdata_from_file.domain
#     # assert experimentdata_continuous._jobs == experimentdata_from_file._jobs


# def test_from_file_wrong_name(experimentdata_continuous: ExperimentData, seed: int, tmp_path: Path):
#     experimentdata_continuous.set_project_dir(tmp_path / 'test001')
#     experimentdata_continuous.store()

#     with pytest.raises(FileNotFoundError):
#         _ = ExperimentData.from_file(tmp_path / 'experimentdata')


# def test_from_sampling(experimentdata_continuous: ExperimentData, seed: int):
#     # sampler = RandomUniform(domain=experimentdata_continuous.domain, number_of_samples=10, seed=seed)
#     experimentdata_from_sampling = ExperimentData.from_sampling(sampler='random',
#                                                                 domain=experimentdata_continuous.domain,
#                                                                 n_samples=10, seed=seed)

#     experimentdata_from_sampling.round(3)
#     assert experimentdata_from_sampling == experimentdata_continuous


# @pytest.fixture
# def sample_csv_inputdata(tmp_path):
#     # Create sample CSV files for testing
#     input_csv_file = tmp_path / 'experimentdata_data.csv'

#     # Create sample input and output dataframes
#     input_data = pd.DataFrame(
#         {'input_col1': [1, 2, 3], 'input_col2': [4, 5, 6]})

#     return input_csv_file, input_data


# @pytest.fixture
# def sample_csv_outputdata(tmp_path):
#     # Create sample CSV files for testing
#     output_csv_file = tmp_path / 'experimentdata_output.csv'

#     # Create sample input and output dataframes
#     output_data = pd.DataFrame(
#         {'output_col1': [7, 8, 9], 'output_col2': [10, 11, 12]})

#     return output_csv_file, output_data


# def test_from_object(experimentdata_continuous: ExperimentData):
#     df_input, df_output = experimentdata_continuous.to_pandas()

#     experiment_data = ExperimentData(
#         input_data=df_input,
#         output_data=df_output,
#         domain=experimentdata_continuous.domain,
#         project_dir=experimentdata_continuous.project_dir)

#     assert experimentdata_continuous == experiment_data

#     # input_data = experimentdata_continuous._input_data
#     # output_data = experimentdata_continuous._output_data
#     # jobs = experimentdata_continuous._jobs
#     # domain = experimentdata_continuous.domain
#     # experiment_data = ExperimentData(
#     #     input_data=input_data, output_data=output_data, jobs=jobs, domain=domain)
#     # assert experiment_data == ExperimentData(
#     #     input_data=input_data, output_data=output_data, jobs=jobs, domain=domain)
#     # assert experiment_data == experimentdata_continuous

# #                                                                              Exporters
# # ======================================================================================


# def test_to_numpy(experimentdata_continuous: ExperimentData, numpy_array: np.ndarray):
#     x, y = experimentdata_continuous.to_numpy()

#     # cast x to floats
#     x = x.astype(float)

#     # assert if x and numpy_array have all the same values
#     assert np.allclose(x, numpy_array, rtol=1e-2)


# def test_to_xarray(experimentdata_continuous: ExperimentData, xarray_dataset: xr.DataSet):
#     exported_dataset = experimentdata_continuous.to_xarray()
#     # assert if xr_dataset is equal to xarray
#     assert exported_dataset.equals(xarray_dataset)


# def test_to_pandas(experimentdata_continuous: ExperimentData, pandas_dataframe: pd.DataFrame):
#     exported_dataframe, _ = experimentdata_continuous.to_pandas()
#     # assert if pandas_dataframe is equal to exported_dataframe
#     pd.testing.assert_frame_equal(
#         exported_dataframe, pandas_dataframe, atol=1e-6, check_dtype=False)
# #                                                                              Exporters
# # ======================================================================================


# def test_set_error(experimentdata_continuous: ExperimentData):
#     experimentdata_continuous.mark(indices=3, status='error')
#     assert experimentdata_continuous.data[3].is_status('error')


# # Helper function to create a temporary CSV file with sample data
# def create_sample_csv_input(file_path):
#     data = [
#         ["x0", "x1", "x2"],
#         [0.77395605, 0.43887844, 0.85859792],
#         [0.69736803, 0.09417735, 0.97562235],
#         [0.7611397, 0.78606431, 0.12811363],
#         [0.45038594, 0.37079802, 0.92676499],
#         [0.64386512, 0.82276161, 0.4434142],
#         [0.22723872, 0.55458479, 0.06381726],
#         [0.82763117, 0.6316644, 0.75808774],
#         [0.35452597, 0.97069802, 0.89312112],
#         [0.7783835, 0.19463871, 0.466721],
#         [0.04380377, 0.15428949, 0.68304895],
#         [0.000000, 0.000000, 0.000000],
#         [1.000000, 1.000000, 1.000000],
#     ]
#     with open(file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerows(data)


# def create_sample_csv_output(file_path):
#     data = [
#         ["y"],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],

#     ]
#     with open(file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerows(data)

# # Pytest fixture to create a temporary CSV file


# def create_domain_pickle(filepath):
#     domain = make_nd_continuous_domain(bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]),
#                                        dimensionality=3)
#     domain.add_output('y', exist_ok=True)
#     domain.store(filepath)


# def create_jobs_csv_finished(filepath):
#     domain = make_nd_continuous_domain(bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]),
#                                        dimensionality=3)

#     _data_input = pd_input()
#     _data_output = pd_output()
#     experimentdata = ExperimentData(
#         domain=domain, input_data=_data_input, output_data=_data_output)
#     experimentdata.jobs.to_csv(Path(filepath).with_suffix('.csv'))


# def create_jobs_csv_open(filepath):
#     domain = make_nd_continuous_domain(bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]),
#                                        dimensionality=3)

#     _data_input = pd_input()
#     experimentdata = ExperimentData(domain=domain, input_data=_data_input)
#     experimentdata.jobs.to_csv(Path(filepath).with_suffix('.csv'))


# def path_domain(tmp_path):
#     domain_file_path = tmp_path / "test_domain.pkl"
#     create_domain_pickle(domain_file_path)
#     return domain_file_path


# def str_domain(tmp_path):
#     domain_file_path = tmp_path / "test_domain.pkl"
#     create_domain_pickle(domain_file_path)
#     return str(domain_file_path)


# def path_jobs_finished(tmp_path):
#     jobs_file_path = tmp_path / "test_jobs.csv"
#     create_jobs_csv_finished(jobs_file_path)
#     return jobs_file_path


# def str_jobs_finished(tmp_path):
#     jobs_file_path = tmp_path / "test_jobs.csv"
#     create_jobs_csv_finished(jobs_file_path)
#     return str(jobs_file_path)


# def path_jobs_open(tmp_path):
#     jobs_file_path = tmp_path / "test_jobs.pkl"
#     create_jobs_csv_open(jobs_file_path)
#     return jobs_file_path


# def str_jobs_open(tmp_path):
#     jobs_file_path = tmp_path / "test_jobs.pkl"
#     create_jobs_csv_open(jobs_file_path)
#     return str(jobs_file_path)


# def path_input(tmp_path):
#     csv_file_path = tmp_path / "test_input.csv"
#     create_sample_csv_input(csv_file_path)
#     return csv_file_path


# def str_input(tmp_path):
#     csv_file_path = tmp_path / "test_input.csv"
#     create_sample_csv_input(csv_file_path)
#     return str(csv_file_path)


# def path_output(tmp_path: Path):
#     csv_file_path = tmp_path / "test_output.csv"
#     create_sample_csv_output(csv_file_path)
#     return csv_file_path


# def str_output(tmp_path: Path):
#     csv_file_path = tmp_path / "test_output.csv"
#     create_sample_csv_output(csv_file_path)
#     return str(csv_file_path)

# # Pytest test function for reading and monkeypatching a CSV file


# def numpy_input(*args, **kwargs):
#     return np.array([
#         [0.77395605, 0.43887844, 0.85859792],
#         [0.69736803, 0.09417735, 0.97562235],
#         [0.7611397, 0.78606431, 0.12811363],
#         [0.45038594, 0.37079802, 0.92676499],
#         [0.64386512, 0.82276161, 0.4434142],
#         [0.22723872, 0.55458479, 0.06381726],
#         [0.82763117, 0.6316644, 0.75808774],
#         [0.35452597, 0.97069802, 0.89312112],
#         [0.7783835, 0.19463871, 0.466721],
#         [0.04380377, 0.15428949, 0.68304895],
#         [0.000000, 0.000000, 0.000000],
#         [1.000000, 1.000000, 1.000000],
#     ])


# def numpy_output(*args, **kwargs):
#     return np.array([
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],

#     ])


# def pd_input(*args, **kwargs):
#     return pd.DataFrame([
#         [0.77395605, 0.43887844, 0.85859792],
#         [0.69736803, 0.09417735, 0.97562235],
#         [0.7611397, 0.78606431, 0.12811363],
#         [0.45038594, 0.37079802, 0.92676499],
#         [0.64386512, 0.82276161, 0.4434142],
#         [0.22723872, 0.55458479, 0.06381726],
#         [0.82763117, 0.6316644, 0.75808774],
#         [0.35452597, 0.97069802, 0.89312112],
#         [0.7783835, 0.19463871, 0.466721],
#         [0.04380377, 0.15428949, 0.68304895],
#         [0.000000, 0.000000, 0.000000],
#         [1.000000, 1.000000, 1.000000],
#     ], columns=["x0", "x1", "x2"])


# def pd_output(*args, **kwargs):
#     return pd.DataFrame([
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],

#     ], columns=["y"])


# # def data_input():
# #     return _Data.from_dataframe(pd_input())


# # def data_output():
# #     return _Data.from_dataframe(pd_output())


# @pytest.mark.parametrize("input_data", [path_input, str_input, pd_input, numpy_input])
# @pytest.mark.parametrize("output_data", [path_output, str_output, pd_output])
# @pytest.mark.parametrize("domain", [
#     make_nd_continuous_domain(bounds=np.array(
#         [[0., 1.], [0., 1.], [0., 1.]]), dimensionality=3),
#     None,
#     path_domain,
#     str_domain
# ])
# @pytest.mark.parametrize("jobs", [None, path_jobs_finished, str_jobs_finished])
# def test_init_with_output(
#     input_data: Union[Callable[[Path], Union[str, Path]], DataTypes],
#     output_data: Union[Callable[[Path], Union[str, Path]], DataTypes],
#     domain: Union[Domain, str, Path, None],
#     jobs: Union[str, Path, None],
#     experimentdata_expected: ExperimentData,
#     monkeypatch,
#     tmp_path: Path,
# ):
#     # Handle callable parameters
#     def resolve_param(param, tmp_path):
#         if callable(param):
#             return param(tmp_path)
#         return param

#     input_data = resolve_param(input_data, tmp_path)
#     output_data = resolve_param(output_data, tmp_path)
#     domain = resolve_param(domain, tmp_path)
#     jobs = resolve_param(jobs, tmp_path)

#     # Mock `pd.read_csv`
#     def mock_read_csv(file_path, *args, **kwargs):
#         path = Path(file_path)
#         if path == tmp_path / "test_input.csv":
#             return experimentdata_expected.to_pandas()[0]
#         elif path == tmp_path / "test_output.csv":
#             return experimentdata_expected.to_pandas()[1]
#         elif path == tmp_path / "test_jobs.csv":
#             return experimentdata_expected.jobs
#         raise ValueError(f"Unexpected file path: {file_path}")

#     # Mock `pickle.load`
#     def mock_load_pickle(file, *args, **kwargs):
#         if Path(file) == tmp_path / "test_domain.pkl":
#             return experimentdata_expected.domain
#         raise ValueError(f"Unexpected pickle file path: {file}")

#     monkeypatch.setattr(pd, "read_csv", mock_read_csv)
#     monkeypatch.setattr(pickle, "load", mock_load_pickle)

#     # # Validation logic for specific inputs
#     # if isinstance(input_data, np.ndarray) and domain is None:
#     #     with pytest.raises(ValueError):
#     #         ExperimentData(domain=domain, input_data=input_data,
#     #                        output_data=output_data, jobs=jobs)
#     #     return

#     # Initialize ExperimentData and validate

#     experiment_data = ExperimentData(
#         domain=domain, input_data=input_data, output_data=output_data,
#         jobs=jobs)
#     experiment_data.domain.add_output('y', exist_ok=True)
#     experiment_data.round(3)

#     print(experiment_data)

#     print(experiment_data.domain)
#     print(experimentdata_expected)

#     print(experimentdata_expected.domain)

#     # Assertions
#     assert experiment_data == experimentdata_expected


# # @pytest.mark.parametrize("input_data", [path_input, str_input, pd_input, numpy_input])
# # @pytest.mark.parametrize("output_data", [path_output, str_output, pd_output])
# # @pytest.mark.parametrize("domain", [make_nd_continuous_domain(bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]),
# #                                                               dimensionality=3), None, path_domain, str_domain])
# # @pytest.mark.parametrize("jobs", [None, path_jobs_finished, str_jobs_finished])
# # def test_init_with_output(
# #         input_data: DataTypes, output_data: DataTypes,
# #         domain: Domain | str | Path | None,
# #         jobs: str | Path | None,
# #         experimentdata_expected: ExperimentData,
# #         monkeypatch, tmp_path: Path):

# #     # if input_data is Callable
# #     if callable(input_data):
# #         input_data = input_data(tmp_path)
# #         expected_data_input = pd.read_csv(input_data)

# #     # if output_data is Callable
# #     if callable(output_data):
# #         output_data = output_data(tmp_path)
# #         expected_data_output = pd.read_csv(output_data)

# #     if callable(domain):
# #         domain = domain(tmp_path)
# #         expected_domain = Domain.from_file(domain)

# #     if callable(jobs):
# #         jobs = jobs(tmp_path)
# #         expected_jobs = pd.read_csv(jobs)

# #     # monkeypatch pd.read_csv to return the expected_data DataFrame
# #     def mock_read_csv(*args, **kwargs):

# #         path = args[0]
# #         if isinstance(args[0], str):
# #             path = Path(path)

# #         if path == tmp_path / "test_input.csv":
# #             return expected_data_input

# #         elif path == tmp_path / "test_output.csv":
# #             return expected_data_output

# #         elif path == tmp_path / "test_jobs.csv":
# #             return expected_jobs

# #         else:
# #             raise ValueError("Unexpected file path")

# #     def mock_load_pickle(*args, **kwargs):
# #         return expected_domain

# #     def mock_pd_read_pickle(*args, **kwargs):
# #         path = args[0]

# #         if isinstance(path, str):
# #             path = Path(path)

# #         if path == tmp_path / "test_jobs.pkl":
# #             return expected_jobs

# #         else:
# #             raise ValueError("Unexpected jobs file path")

# #     monkeypatch.setattr(pd, "read_csv", mock_read_csv)
# #     monkeypatch.setattr(pickle, "load", mock_load_pickle)
# #     monkeypatch.setattr(pd, "read_pickle", mock_pd_read_pickle)

# #     if isinstance(input_data, np.ndarray) and domain is None:
# #         with pytest.raises(ValueError):
# #             ExperimentData(domain=domain, input_data=input_data,
# #                            output_data=output_data, jobs=jobs)
# #         return
# #     # Initialize ExperimentData with the CSV file
# #     experiment_data = ExperimentData(domain=domain, input_data=input_data,
# #                                      output_data=output_data, jobs=jobs)

# #     experiment_data.round(3)

# #     # Check if the input_data attribute of ExperimentData matches the expected_data
# #     # pd.testing.assert_frame_equal(
# #     #     experiment_data._input_data.to_dataframe(), experimentdata_expected._input_data.to_dataframe(), check_dtype=False, atol=1e-6)
# #     # pd.testing.assert_frame_equal(experiment_data._output_data.to_dataframe(),
# #     #                               experimentdata_expected._output_data.to_dataframe(), check_dtype=False)

# #     assert experiment_data == experimentdata_expected


# # @pytest.mark.parametrize("input_data", [pd_input(), path_input, str_input, numpy_input()])
# # @pytest.mark.parametrize("output_data", [None])
# # @pytest.mark.parametrize("domain", [make_nd_continuous_domain(bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]),
# #                                                               dimensionality=3), None, path_domain, str_domain])
# # @pytest.mark.parametrize("jobs", [None, path_jobs_open, str_jobs_open])
# # def test_init_without_output(input_data: DataTypes, output_data: DataTypes, domain: Domain, jobs: _JobQueue,
# #                              experimentdata_expected_no_output: ExperimentData, monkeypatch, tmp_path):

# #     # if input_data is Callable
# #     if callable(input_data):
# #         input_data = input_data(tmp_path)
# #         expected_data_input = pd.read_csv(input_data)

# #     # if output_data is Callable
# #     if callable(output_data):
# #         output_data = output_data(tmp_path)
# #         expected_data_output = pd.read_csv(output_data)

# #     if callable(domain):
# #         domain = domain(tmp_path)
# #         expected_domain = Domain.from_file(domain)

# #     if callable(jobs):
# #         jobs = jobs(tmp_path)
# #         expected_jobs = _JobQueue.from_file(jobs).jobs

# #     # monkeypatch pd.read_csv to return the expected_data DataFrame
# #     def mock_read_csv(*args, **kwargs):

# #         path = args[0]
# #         if isinstance(args[0], str):
# #             path = Path(path)

# #         if path == tmp_path / "test_input.csv":
# #             return expected_data_input

# #         elif path == tmp_path / "test_output.csv":
# #             return expected_data_output

# #         else:
# #             raise ValueError("Unexpected file path")

# #     def mock_load_pickle(*args, **kwargs):
# #         return expected_domain

# #     def mock_pd_read_pickle(*args, **kwargs):
# #         path = args[0]

# #         if isinstance(path, str):
# #             path = Path(path)

# #         if path == tmp_path / "test_jobs.pkl":
# #             return expected_jobs

# #     monkeypatch.setattr(pd, "read_csv", mock_read_csv)
# #     monkeypatch.setattr(pickle, "load", mock_load_pickle)
# #     monkeypatch.setattr(pd, "read_pickle", mock_pd_read_pickle)

# #     if isinstance(input_data, np.ndarray) and domain is None:
# #         with pytest.raises(ValueError):
# #             ExperimentData(domain=domain, input_data=input_data,
# #                            output_data=output_data, jobs=jobs)
# #         return

# #     # Initialize ExperimentData with the CSV file
# #     experiment_data = ExperimentData(domain=domain, input_data=input_data,
# #                                      output_data=output_data, jobs=jobs)

# #     # Check if the input_data attribute of ExperimentData matches the expected_data
# #     pd.testing.assert_frame_equal(
# #         experiment_data._input_data.to_dataframe(), experimentdata_expected_no_output._input_data.to_dataframe(), atol=1e-6, check_dtype=False)
# #     pd.testing.assert_frame_equal(experiment_data._output_data.to_dataframe(),
# #                                   experimentdata_expected_no_output._output_data.to_dataframe())
# #     pd.testing.assert_series_equal(
# #         experiment_data._jobs.jobs, experimentdata_expected_no_output._jobs.jobs)
# #     # assert experiment_data.domain == experimentdata_expected_no_output.domain
# #     assert experiment_data._jobs == experimentdata_expected_no_output._jobs


# # @pytest.mark.parametrize("input_data", [None])
# # @pytest.mark.parametrize("output_data", [None])
# # @pytest.mark.parametrize("domain", [make_nd_continuous_domain(bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]),
# #                                                               dimensionality=3), path_domain, str_domain])
# # def test_init_only_domain(input_data: DataTypes, output_data: DataTypes, domain: Domain | str | Path,
# #                           experimentdata_expected_only_domain: ExperimentData,
# #                           monkeypatch, tmp_path):

# #     # if input_data is Callable
# #     if callable(input_data):
# #         input_data = input_data(tmp_path)
# #         expected_data_input = pd.read_csv(input_data)

# #     # if output_data is Callable
# #     if callable(output_data):
# #         output_data = output_data(tmp_path)
# #         expected_data_output = pd.read_csv(output_data)

# #     if callable(domain):
# #         domain = domain(tmp_path)
# #         expected_domain = Domain.from_file(domain)

# #     # monkeypatch pd.read_csv to return the expected_data DataFrame
# #     def mock_read_csv(*args, **kwargs):

# #         path = args[0]
# #         if isinstance(args[0], str):
# #             path = Path(path)

# #         if path == tmp_path / "test_input.csv":
# #             return expected_data_input

# #         elif path == tmp_path / "test_output.csv":
# #             return expected_data_output

# #         else:
# #             raise ValueError("Unexpected file path")

# #     def mock_load_pickle(*args, **kwargs):
# #         return expected_domain

# #     monkeypatch.setattr(pd, "read_csv", mock_read_csv)
# #     monkeypatch.setattr(pickle, "load", mock_load_pickle)

# #     # Initialize ExperimentData with the CSV file
# #     experiment_data = ExperimentData(domain=domain, input_data=input_data,
# #                                      output_data=output_data)

# #     # Check if the input_data attribute of ExperimentData matches the expected_data
# #     pd.testing.assert_frame_equal(
# #         experiment_data._input_data.to_dataframe(), experimentdata_expected_only_domain._input_data.to_dataframe(), check_dtype=False)
# #     pd.testing.assert_frame_equal(experiment_data._output_data.to_dataframe(),
# #                                   experimentdata_expected_only_domain._output_data.to_dataframe(), check_dtype=False)
# #     assert experiment_data._input_data == experimentdata_expected_only_domain._input_data
# #     assert experiment_data._output_data == experimentdata_expected_only_domain._output_data
# #     assert experiment_data.domain == experimentdata_expected_only_domain.domain
# #     assert experiment_data._jobs == experimentdata_expected_only_domain._jobs

# #     assert experiment_data == experimentdata_expected_only_domain


# @pytest.mark.parametrize("input_data", [[0.1, 0.2], {"a": 0.1, "b": 0.2}, 0.2, 2])
# def test_invalid_type(input_data):
#     with pytest.raises(TypeError):
#         ExperimentData(input_data=input_data)


# def test_add_invalid_type(experimentdata: ExperimentData):
#     with pytest.raises(TypeError):
#         experimentdata + 1


# def test_add_two_different_domains(experimentdata: ExperimentData, experimentdata_continuous: ExperimentData):
#     with pytest.raises(ValueError):
#         experimentdata + experimentdata_continuous


# def test_repr_html(experimentdata: ExperimentData, monkeypatch):
#     assert isinstance(experimentdata._repr_html_(), str)


# def test_store(experimentdata: ExperimentData, tmp_path: Path):
#     experimentdata.store(tmp_path / "test")
#     assert (tmp_path / "test" / "experiment_data" / "input.csv").exists()
#     assert (tmp_path / "test" / "experiment_data" / "output.csv").exists()
#     assert (tmp_path / "test" / "experiment_data" / "domain.pkl").exists()
#     assert (tmp_path / "test" / "experiment_data" / "jobs.pkl").exists()


# def test_store_give_no_filename(experimentdata: ExperimentData, tmp_path: Path):
#     experimentdata.set_project_dir(tmp_path / 'test2')
#     experimentdata.store()
#     assert (tmp_path / "test2" / "experiment_data" / "input.csv").exists()
#     assert (tmp_path / "test2" / "experiment_data" / "output.csv").exists()
#     assert (tmp_path / "test2" / "experiment_data" / "domain.pkl").exists()
#     assert (tmp_path / "test2" / "experiment_data" / "jobs.pkl").exists()


# @pytest.mark.parametrize("mode", ["sequential", "parallel", "typo"])
# def test_evaluate_mode(mode: str, experimentdata_continuous: ExperimentData, tmp_path: Path):
#     experimentdata_continuous.filename = tmp_path / 'test009'

#     if mode == "typo":
#         with pytest.raises(ValueError):
#             experimentdata_continuous.evaluate(
#                 data_generator="ackley", mode=mode,
#                 scale_bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]),
#                 seed=SEED)
#     else:
#         experimentdata_continuous.evaluate(
#             data_generator="ackley", mode=mode,
#             scale_bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]),
#             seed=SEED)


# @pytest.mark.parametrize("selection", ["x0", ["x0"], ["x0", "x2"]])
# def test_get_input_data_selection(experimentdata_expected_no_output: ExperimentData, selection: Iterable[str] | str):
#     input_data = experimentdata_expected_no_output.get_input_data(selection)
#     df, _ = input_data.to_pandas()
#     if isinstance(selection, str):
#         selection = [selection]
#     selected_pd = pd_input()[selection]
#     pd.testing.assert_frame_equal(
#         df, selected_pd, check_dtype=False, atol=1e-6)


# def test_iter_behaviour(experimentdata_continuous: ExperimentData):
#     for i in experimentdata_continuous:
#         assert isinstance(i, ExperimentSample)

#     selected_experimentdata = experimentdata_continuous.select([0, 2, 4])
#     for i in selected_experimentdata:
#         assert isinstance(i, ExperimentSample)


# def test_select_with_status_open(experimentdata: ExperimentData):
#     selected_data = experimentdata.select_with_status('open')
#     assert all(es.is_status('open') for _, es in selected_data)


# def test_select_with_status_in_progress(experimentdata: ExperimentData):
#     selected_data = experimentdata.select_with_status('in progress')
#     assert all(es.is_status('in progress') for _, es in selected_data)


# def test_select_with_status_finished(experimentdata: ExperimentData):
#     selected_data = experimentdata.select_with_status('finished')
#     assert all(es.is_status('finished') for _, es in selected_data)


# def test_select_with_status_error(experimentdata: ExperimentData):
#     selected_data = experimentdata.select_with_status('error')
#     assert all(es.is_status('error') for _, es in selected_data)


# def test_select_with_status_invalid_status(experimentdata: ExperimentData):
#     with pytest.raises(ValueError):
#         _ = experimentdata.select_with_status('invalid_status')


# if __name__ == "__main__":  # pragma: no cover
#     pytest.main()
