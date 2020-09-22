'''
Created on 2020-04-22 19:50:46
Last modified on 2020-09-22 07:43:42
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)
'''


# imports

# standard library
import os
from collections import OrderedDict
import time
import pickle
import multiprocessing as mp
import traceback
import shutil

# third party
import numpy as np

# local library
from f3das.utils.file_handling import verify_existing_name
from f3das.run.utils import get_updated_sims_state


# TODO: interrupt simulations


# run abaqus

def run_sims(example_name, n_sims=None, n_cpus=1,
             points=None, pkl_filename='DoE.pkl', sims_dir_name='analyses',
             run_module_name='f3das.abaqus.run.run_model',
             keep_odb=True, dump_py_objs=False, abaqus_path='abaqus',
             gui=False):
    '''
    IMPORTANT: if number cpus>1 (parallel simulations, not simulation in
    parallel), function must be inside "if __name__='__main__':" in the script.

    Parameters
    ----------
    n_sims : int or None
        Number of simulations to run. Ignored if 'points' is not None.
        'missing_sims' in the main file are considered here. Runs all the
        missing simulations if None. Order of 'missing_sims' is considered.
    n_cpus : int
        Number of simultaneous processes. If job's 'n_cpus'>1, it is automatically
        set to 1 (for now, it is not possible to run several multi-process
        simulations simultaneously).
    points : array or None
        DoE points to run. If None, 'n_sims' are run. Simulations with
        folders already created are run again (so, be careful!).
    pkl_filename : str
        Main file name.
    keep_odb : bool
        Keep odb after simulation? If yes, make sure you have enough storage.
    dump_py_objs : bool
        Store Python objects that were used to create and run the numerical
        simulations? If yes, a file with extension 'pkl_abq' is created.
        Specially useful for debugging.
    '''

    # TODO: possibility of zipping odb
    # TODO: verify licences

    # create analyses dir
    dir_full_path = os.path.join(example_name, sims_dir_name)
    if not os.path.exists(dir_full_path):
        os.mkdir(dir_full_path)

    # get data
    with open(os.path.join(example_name, pkl_filename), 'rb') as file:
        data = pickle.load(file)

    # points or n_sims?
    if points is not None:
        for key, value in data['run_info'].items():
            if key != 'running_sims':
                data['run_info'][key] = [point for point in value if point not in points]
    else:
        n_sims = len(data['run_info']['missing_sims']) if n_sims is None else n_sims
        points = data['run_info']['missing_sims'][:n_sims]
        data['run_info']['missing_sims'] = data['run_info']['missing_sims'][n_sims:]

    # update data temporarily
    data['run_info']['running_sims'].extend(points)
    with open(os.path.join(example_name, pkl_filename), 'wb') as file:
        pickle.dump(data, file)

    try:

        # run in parallel?
        sim_info = data['sim_info']['sim_info']
        n_cpus_sim = np.array([sim['job_info'].get('n_cpus', 1) for sim in sim_info.values()])
        n_cpus = 1 if np.prod(n_cpus_sim) != 1 else n_cpus

        # create pkl for each doe
        _create_DoE_sim_info(example_name, points, sims_dir_name=sims_dir_name,
                             pkl_filename=pkl_filename, keep_odb=keep_odb,
                             dump_py_objs=dump_py_objs)

        # run
        if n_cpus > 1:

            # distribute points
            points = sorted(points)
            points_cpus = []
            for i in range(n_cpus):
                points_cpus.append(points[i::n_cpus])

            # start pool
            pool = mp.Pool(n_cpus)

            # run sims
            for i, points in enumerate(points_cpus):
                wait_time = i * 5
                pool.apply_async(_run_sims_sequentially,
                                 args=(example_name, points, wait_time,
                                       run_module_name, sims_dir_name,
                                       abaqus_path, gui))
            # close pool and wait process completion
            pool.close()
            pool.join()

        else:
            _run_sims_sequentially(example_name, points,
                                   run_module_name=run_module_name,
                                   sims_dir_name=sims_dir_name,
                                   abaqus_path=abaqus_path, gui=gui)
    except:
        traceback.print_exc()

    finally:
        # based on points, reupdate data['run_info']
        error_sims_, successful_sims_ = get_updated_sims_state(
            example_name, points, sims_dir_name)

        points_ = list(set(points) - set(error_sims_) - set(successful_sims_))
        data['run_info']['missing_sims'].extend(points_)
        data['run_info']['missing_sims'].sort()

        data['run_info']['error_sims'].extend(error_sims_)
        data['run_info']['error_sims'].sort()

        data['run_info']['successful_sims'].extend(successful_sims_)
        data['run_info']['successful_sims'].sort()

        running_sims = data['run_info']['running_sims']
        data['run_info']['running_sims'] = sorted(list(set(running_sims).difference(set(points))))

        with open(os.path.join(example_name, pkl_filename), 'wb') as file:
            pickle.dump(data, file)


def _create_DoE_sim_info(example_name, points, sims_dir_name='analyses',
                         pkl_filename='DoE.pkl', keep_odb=True,
                         dump_py_objs=False,):

    # get data
    with open(os.path.join(example_name, pkl_filename), 'rb') as file:
        data = pickle.load(file)
    doe_variables = data['doe_variables']
    datapoints = data['points']
    sim_info = data['sim_info']
    transform_inputs = sim_info.get('transform_inputs', None)
    fixed_variables = data.get('fixed_variables', {})
    additional_variables = data.get('additional_variables', {})

    # variables to save
    abstract_model = sim_info['abstract_model']
    post_processing_fnc = sim_info.get('post_processing_fnc', None)

    # deal with subroutines
    subroutine_names = []
    for sim_info_ in sim_info['sim_info'].values():
        subroutine_name = sim_info_['job_info'].get('userSubroutine', None)
        if subroutine_name:
            subroutine_loc_ls = subroutine_name.split('.')
            subroutine_loc = '{}.{}'.format(os.path.join(*subroutine_loc_ls[:-1]), subroutine_loc_ls[-1])
            subroutine_names.append((subroutine_loc, '.'.join(subroutine_loc_ls[-2::])))
            sim_info_['job_info']['userSubroutine'] = subroutine_names[-1][1]

    # create pkl files
    dir_full_path = os.path.join(example_name, sims_dir_name)
    for point in points:
        doe_dir_name = os.path.join(dir_full_path, 'DoE_point{}'.format(point))
        if os.path.exists(doe_dir_name):
            shutil.rmtree(doe_dir_name)
        os.mkdir(doe_dir_name)

        # dict with all the variables
        variables = datapoints.loc[point, doe_variables.keys()].to_dict()
        for key, value in variables.items():
            if type(value) is np.float64:
                variables[key] = float(value)
        variables.update(fixed_variables)
        for key, value in additional_variables.items():
            variables[key] = float(value[point])

        # if required, transform inputs
        if callable(transform_inputs):
            variables = transform_inputs(variables)

        # create and dump dict
        data = OrderedDict({'abstract_model': abstract_model,
                            'post_processing_fnc': post_processing_fnc,
                            'variables': variables,
                            'sim_info': sim_info['sim_info'],
                            'keep_odb': keep_odb,
                            'dump_py_objs': dump_py_objs,
                            'success': None})

        with open(os.path.join(doe_dir_name, 'sim.pkl'), 'wb') as file:
            pickle.dump(data, file, protocol=2)

        # copy subroutine
        if subroutine_names:
            for subroutine_name in subroutine_names:
                shutil.copyfile(subroutine_name[0], os.path.join(doe_dir_name, subroutine_name[1]))


def _run_sims_sequentially(example_name, points, wait_time=0,
                           run_module_name='f3das.abaqus.run.run_model',
                           sims_dir_name='analyses', abaqus_path='abaqus',
                           gui=False):
    '''

    Parameters
    ----------
    # TODO: change docstrings

    '''

    # initialization
    time.sleep(wait_time)

    # create run filename
    run_filename = verify_existing_name('_temp.py')
    lines = ['import runpy',
             'import os',
             'import sys',
             'initial_wd = os.getcwd()',
             'sys.path.append(initial_wd)',
             'points = %s' % points,
             "sim_dir = r'%s'" % os.path.join(example_name, sims_dir_name),
             'for point in points:',
             "\tos.chdir('%s' % os.path.join(sim_dir, 'DoE_point%i' % point))",
             "\trunpy.run_module('%s', run_name='__main__')" % run_module_name,
             '\tos.chdir(initial_wd)']
    with open(run_filename, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    # open abaqus and run module
    gui_ = 'script' if gui else 'noGUI'
    command = '{} cae {}={}'.format(abaqus_path, gui_, run_filename)
    os.system(command)

    # clear temporary run file
    os.remove(run_filename)


# function definition

def convert_dict_unicode_str(pickled_dict):
    new_dict = OrderedDict() if type(pickled_dict) is OrderedDict else {}
    for key, value in pickled_dict.items():
        value = _set_converter_flow(value)
        new_dict[str(key)] = value

    return new_dict


def convert_iterable_unicode_str(iterable):
    new_iterable = []
    for value in iterable:
        value = _set_converter_flow(value)
        new_iterable.append(value)

    if type(iterable) is tuple:
        new_iterable = tuple(new_iterable)
    elif type(iterable) is set:
        new_iterable = set(new_iterable)

    return new_iterable


def _set_converter_flow(value):

    if type(value) is unicode:
        value = str(value)
    elif type(value) in [OrderedDict, dict]:
        value = convert_dict_unicode_str(value)
    elif type(value) in [list, tuple, set]:
        value = convert_iterable_unicode_str(value)

    return value
