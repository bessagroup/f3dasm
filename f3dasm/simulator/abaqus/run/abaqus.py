'''
Created on 2020-04-22 19:50:46
Last modified on 2020-09-30 11:41:45

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
import f3dasm
from .utils import get_updated_sims_state
from ..utils.file_handling import verify_existing_name
from ..utils.utils import import_abstract_obj
from ..post_processing import post_process_sims
from ..post_processing import concatenate_raw_data
from ..post_processing import collect_raw_data


# TODO: interrupt simulations


# run abaqus

def run_sims(example_name, n_sims=None, n_cpus=1, points=None,
             data_filename='DoE.pkl', raw_data_filename='raw_data.pkl',
             sims_dir_name='analyses', run_module_name='f3dasm.abaqus.run.run_model',
             keep_odb=True, dump_py_objs=False, abaqus_path='abaqus',
             gui=False, delete=False, pp_fnc=None, pp_fnc_kwargs=None,
             create_new_file='',):
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
    data_filename : str
        Main file name.
    keep_odb : bool
        Keep odb after simulation? If yes, make sure you have enough storage.
    dump_py_objs : bool
        Store Python objects that were used to create and run the numerical
        simulations? If yes, a file with extension 'pkl_abq' is created.
        Specially useful for debugging.

    # TODO: complete docstrings
    '''

    # TODO: possibility of zipping odb
    # TODO: verify licences

    # get data
    with open(os.path.join(example_name, data_filename), 'rb') as file:
        data = pickle.load(file)

    # points or n_sims?
    if points is not None:
        points = set(points)
        for key, value in data['run_info'].items():
            if key != 'running_sims':
                value.difference_update(points)
    else:
        n_sims = len(data['run_info']['missing_sims']) if n_sims is None else n_sims
        points = set(list(data['run_info']['missing_sims'])[:n_sims])
        data['run_info']['missing_sims'].difference_update(points)
    data['run_info']['running_sims'].update(points)

    # create analyses dir
    dir_full_path = os.path.join(example_name, sims_dir_name)
    if not os.path.exists(dir_full_path):
        os.mkdir(dir_full_path)

    # create pkl for each doe
    _create_DoE_sim_info(example_name, data, points, sims_dir_name=sims_dir_name,
                         keep_odb=keep_odb, dump_py_objs=dump_py_objs,)

    # run in parallel? (restriction due to no control of cpu allocation)
    sim_info = data['sim_info']
    n_cpus_sim = np.array([sim['job_info'].get('n_cpus', 1) for sim in sim_info.values()])
    n_cpus = 1 if np.prod(n_cpus_sim) != 1 else n_cpus

    # create _temp folder and copy f3dasm
    temp_dir_name = '_temp'
    _create_temp_dir(temp_dir_name)

    # update data temporarily (due to run_info)
    with open(os.path.join(example_name, data_filename), 'wb') as file:
        pickle.dump(data, file)

    try:
        # TODO: find a way to verify license and don't use try
        # run
        if n_cpus > 1:
            _run_sims_in_parallel(example_name, points, n_cpus,
                                  run_module_name=run_module_name,
                                  sims_dir_name=sims_dir_name,
                                  abaqus_path=abaqus_path, gui=gui,
                                  temp_dir_name=temp_dir_name)

        else:
            _run_sims_sequentially(example_name, points,
                                   run_module_name=run_module_name,
                                   sims_dir_name=sims_dir_name,
                                   abaqus_path=abaqus_path, gui=gui,
                                   temp_dir_name=temp_dir_name)
    except:
        traceback.print_exc()

    finally:

        # delete _temp dir
        shutil.rmtree(temp_dir_name)

        # concatenate (and/or collect) data
        if raw_data_filename:
            raw_data = concatenate_raw_data(
                example_name, data_filename=data_filename,
                raw_data_filename=raw_data_filename, sims_dir_name=sims_dir_name,
                delete=delete, compress=True, sim_numbers=points).loc[points]
        else:
            raw_data = collect_raw_data(example_name, sims_dir_name=sims_dir_name,
                                        delete=False, raw_data_filename='',
                                        sim_numbers=points)

        # update sims state
        successful_sims = _update_sims_state(data, points, raw_data)

        # automatic post-processing
        if pp_fnc is not None and len(successful_sims):
            # TODO: variant of pp_fnc when user defines variable; how to have both ways of defining variables?
            data = post_process_sims(pp_fnc, example_name,
                                     sim_numbers=successful_sims,
                                     data_filename='', data=data,
                                     raw_data=raw_data.loc[successful_sims],
                                     pp_fnc_kwargs=pp_fnc_kwargs,
                                     create_new_file=create_new_file)

        # store data with updated `run_info`
        with open(os.path.join(example_name, data_filename), 'wb') as file:
            pickle.dump(data, file)


def _create_DoE_sim_info(example_name, data, points, sims_dir_name='analyses',
                         keep_odb=True, dump_py_objs=False,):
    # get data
    doe_variables = data['doe_variables']
    datapoints = data['points']
    sim_info = data['sim_info']
    transform_inputs = data.get('transform_inputs', None)
    fixed_variables = data.get('fixed_variables', {})
    additional_variables = data.get('additional_variables', {})

    # deal with subroutines
    copy_subroutines_fncs = _get_copy_subroutines(sim_info)

    # manipulate transform inputs
    if transform_inputs is not None:
        transform_inputs = import_abstract_obj(transform_inputs)

    # create pkl files
    dir_full_path = os.path.join(example_name, sims_dir_name)
    for point in points:
        doe_dir_name = os.path.join(dir_full_path, 'DoE_point{}'.format(point))
        if os.path.exists(doe_dir_name):
            shutil.rmtree(doe_dir_name)
        os.mkdir(doe_dir_name)

        # dict with all the variables
        variables = datapoints.loc[point, doe_variables.keys()].to_dict()
        variables.update(fixed_variables)
        for key, value in additional_variables.items():
            variables[key] = value[point]
        for key, value in variables.items():
            if type(value) is np.float64:
                variables[key] = float(value)

        # if required, transform inputs
        if transform_inputs is not None:
            variables = transform_inputs(variables)

        # create and dump dict
        data = OrderedDict({'variables': variables,
                            'sim_info': sim_info,
                            'keep_odb': keep_odb,
                            'dump_py_objs': dump_py_objs, })

        with open(os.path.join(doe_dir_name, 'sim.pkl'), 'wb') as file:
            pickle.dump(data, file, protocol=2)

        # copy subroutine
        for copy_subroutines_fnc in copy_subroutines_fncs:
            copy_subroutines_fnc(doe_dir_name)


def _run_sims_sequentially(example_name, points, wait_time=0,
                           run_module_name='f3dasm.abaqus.run.run_model',
                           sims_dir_name='analyses', abaqus_path='abaqus',
                           gui=False, temp_dir_name='_temp'):
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
             "sys.path.append(os.path.join(initial_wd, '%s'))" % temp_dir_name,
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


def _run_sims_in_parallel(example_name, points, n_cpus,
                          run_module_name='f3dasm.abaqus.run.run_model',
                          sims_dir_name='analyses', abaqus_path='abaqus',
                          gui=False, temp_dir_name='_temp'):

    # distribute points
    points = sorted(points)
    points_cpus = []
    for i in range(n_cpus):
        points_cpus.append(points[i::n_cpus])

    # start pool
    pool = mp.Pool(n_cpus)

    # run sims
    # TODO: pool reserve cpus?
    for i, points in enumerate(points_cpus):
        wait_time = i * 5
        pool.apply_async(_run_sims_sequentially,
                         args=(example_name, points, wait_time,
                               run_module_name, sims_dir_name,
                               abaqus_path, gui, temp_dir_name))
    # close pool and wait process completion
    pool.close()
    pool.join()


def _create_temp_dir(temp_dir_name='_temp'):
    if not os.path.exists(temp_dir_name):
        os.mkdir(temp_dir_name)
    new_f3das_dir = os.path.join(temp_dir_name, 'f3dasm')
    if os.path.exists(new_f3das_dir):
        shutil.rmtree(new_f3das_dir)
    shutil.copytree(f3dasm.__path__[0], new_f3das_dir)


def _update_sims_state(data, points, raw_data):

    # based on points, reupdate data['run_info']
    error_sims_, successful_sims_ = get_updated_sims_state(raw_data=raw_data)

    points_ = points - error_sims_ - successful_sims_
    data['run_info']['missing_sims'].update(points_)
    data['run_info']['error_sims'].update(error_sims_)
    data['run_info']['successful_sims'].update(successful_sims_)
    data['run_info']['running_sims'].difference_update(points)

    return successful_sims_


def _get_copy_subroutines(sim_info):
    # input DoE dir name
    subroutine_copy_fncs = []
    for sim_info_ in sim_info.values():
        subroutine_name = sim_info_['job_info'].get('userSubroutine', None)
        if subroutine_name:
            subroutine_loc_ls = subroutine_name.split('.')
            subroutine_loc = '{}.{}'.format(os.path.join(*subroutine_loc_ls[:-1]), subroutine_loc_ls[-1])
            subroutine_name = '.'.join(subroutine_loc_ls[-2::])
            sim_info_['job_info']['userSubroutine'] = subroutine_name

            def fnc(doe_dir_name, subroutine_loc=subroutine_loc, subroutine_name=subroutine_name):
                return shutil.copy(subroutine_loc, os.path.join(doe_dir_name, subroutine_name))
        else:
            def fnc(doe_dir_name): pass
        subroutine_copy_fncs.append(fnc)

    return subroutine_copy_fncs
