'''
Created on 2020-05-05 16:28:59
Last modified on 2020-09-30 11:42:01

@author: L. F. Pereira (lfpereira@fe.up.pt)
'''


# imports

# standard library
import os
import pickle
import gzip
import shutil

# third-party
import pandas as pd

# local library
from .utils.file_handling import get_unique_file_by_ext
from .utils.file_handling import collect_folder_names
from .utils.utils import get_int_number_from_str
from .utils.utils import read_pkl_file

# TODO: possibility of entering abaqus for post-processing


# function definition

def post_process_sims(pp_fnc, example_name, sim_numbers=None,
                      sims_dir_name='analyses', data_filename='DoE.pkl',
                      data=None, raw_data='raw_data.pkl', create_new_file='',
                      pp_fnc_kwargs=None):
    '''
    Parameters
    ----------
    create_new_file : str
        If not empty, then a new file is created containing the information
        in `data_filename` plus the treated outputs.
    data_filename : str
        If `data` is given and `data_filename` is empty, then an updated file
        will not be stored.
    data : dict
        If given, `data_filename` will be ignored during data reading.
    raw_data : pd.Series or str or None.
        Data is gatherer according to `raw_data` type. The possibilities are:
            None: simulation folders
            str: raw data file
            pandas.Series: uses itself.

    Notes
    -----
    1. If the output variables already exist in 'points', their values are
    updated only if the post-processing data for the given point is available
    in `sims_dir_name`. Otherwise, older values are kept.
    '''

    # initialization
    pp_fnc_kwargs = {} if pp_fnc_kwargs is None else pp_fnc_kwargs

    # get current pandas
    if data is None:
        with open(os.path.join(example_name, data), 'rb') as file:
            data = pickle.load(file)
    points = data['points']

    # get available simulations
    if type(raw_data) is str or raw_data is None:
        raw_data = collect_raw_data(example_name, sims_dir_name=sims_dir_name,
                                    sim_numbers=sim_numbers, delete=False,
                                    raw_data_filename=raw_data)

    # add outputs to pd.Dataframe (also post-processes 1st simulation)
    column_names = list(points.columns.values)
    sim_numbers = list(raw_data.keys()) if sim_numbers is None else list(sim_numbers)
    sim_number = sim_numbers.pop(0)
    data_sim = raw_data.loc[sim_number]
    results_sim = pp_fnc(data_sim, **pp_fnc_kwargs)
    output_variables = list(results_sim.keys())
    for variable in output_variables[::-1]:
        if variable not in column_names:
            points.insert(loc=len(column_names), value=None, column=variable)
    for key, value in results_sim.items():
        points.loc[sim_number, key] = value

    # get results for each simulation
    for sim_number, data_sim in raw_data.loc[sim_numbers].iteritems():
        results_sim = pp_fnc(data_sim, **pp_fnc_kwargs)
        for key, value in results_sim.items():
            points.loc[sim_number, key] = value

    # create new pickle file
    data_filename_output = create_new_file if create_new_file else data_filename
    if data_filename_output:
        with open(os.path.join(example_name, data_filename_output), 'wb') as file:
            pickle.dump(data, file)

    return data


def get_sim_data(dir_name, folder_name):

    # TODO: add possibility of using gzip

    # get filename and read data
    dir_name = os.path.join(dir_name, folder_name)
    filename = get_unique_file_by_ext(dir_name, ext='.pkl')

    with open(os.path.join(dir_name, filename), 'rb') as file:
        data = pickle.load(file, encoding='latin1')

    return data


def concatenate_raw_data(example_name, data_filename='DoE.pkl',
                         raw_data_filename='raw_data.pkl', sims_dir_name='analyses',
                         delete=False, compress=True, sim_numbers=None):
    '''
    Creates an unique file that contains all the information of the problem.

    Parameters
    ----------
    compress : bool
        If True, zips file using gzip. It may take longer. Depending on the data,
        the compression ratio may be huge.

    Notes
    -----
    1. If file already exists, then data in 'analyses' is added to the
    already existing information (overriding pre-existing information).
    '''

    # initialization
    open_file = gzip.open if compress else open

    # verify if file already exists
    if os.path.exists(os.path.join(example_name, raw_data_filename)):
        raw_data = read_pkl_file(os.path.join(example_name, raw_data_filename))
    else:
        raw_data = pd.Series(dtype=object)

    # get available simulations
    new_raw_data = collect_raw_data_from_folders(example_name,
                                                 sims_dir_name=sims_dir_name,
                                                 sim_numbers=sim_numbers, delete=delete)
    raw_data = new_raw_data.combine_first(raw_data).sort_index()

    # save file
    with open_file(os.path.join(example_name, raw_data_filename), 'wb') as file:
        pickle.dump(raw_data, file)

    return raw_data


def collect_raw_data(example_name, sims_dir_name='analyses', sim_numbers=None,
                     delete=False, raw_data_filename='raw_data.pkl'):
    if raw_data_filename:
        raw_data = read_pkl_file(os.path.join(example_name, raw_data_filename))
        if sim_numbers is not None:
            raw_data = raw_data.loc[sim_numbers]
    else:
        raw_data = collect_raw_data_from_folders(
            example_name, sims_dir_name=sims_dir_name,
            sim_numbers=sim_numbers, delete=False)

    return raw_data


def collect_raw_data_from_folders(example_name, sims_dir_name='analyses',
                                  sim_numbers=None, delete=False):

    # get available simulations
    dir_name = os.path.join(example_name, sims_dir_name)
    folder_names = collect_folder_names(dir_name, sim_numbers=sim_numbers)

    # store the information of each available DoE point
    raw_data = {}
    for folder_name in folder_names:
        # simulation number
        i = get_int_number_from_str(folder_name)

        # get data
        raw_data[i] = get_sim_data(dir_name, folder_name)

        # delete folder
        if delete:
            shutil.rmtree(os.path.join(dir_name, folder_name))

    return pd.Series(raw_data)
