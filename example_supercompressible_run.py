'''
Created on 2020-04-22 14:43:42
Last modified on 2020-05-11 19:43:40
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Show how the code to run one DoE point works.
'''


#%% imports

# standard library
import os
from collections import OrderedDict
import pickle

# local library
from f3das.misc.file_handling import verify_existing_name
from f3das.design_of_experiments.transform_inputs import transform_inputs_supercompressible


#%% initialization

gui = False
dir_name = os.path.join(os.getcwd(), verify_existing_name('test'))
filename = 'simul.pkl'

# variable definition
inputs_type = 'normal'  # choose inputs
variables = {'normal': {'n_longerons': 3,
                        'bottom_diameter': 100.,
                        'young_modulus': 3500.0,
                        'top_diameter': 83.75,
                        'pitch': 120.703125,
                        'shear_modulus': 1287.0,
                        'Ixx': 14.891743763069467,
                        'Iyy': 14.891743763069467,
                        'J': 29.783487526138934,
                        'area': 13.679735787682546,
                        'imperfection': 0.052175599251571225,
                        'section': 'generalized'},
             'ratio': {'n_longerons': 3,
                       'bottom_diameter': 100.,
                       'young_modulus': 1826.,
                       'ratio_area': 0.001,
                       'ratio_shear_modulus': .36,
                       'ratio_Ixx': 7.5e-7,
                       'ratio_Iyy': 1e-6,
                       'ratio_J': 2e-6,
                       'ratio_pitch': .66,
                       'ratio_top_diameter': 4.72e-6,
                       'imperfection': 7.85114e-2,
                       'section': 'generalized'},
             'circular': {'n_longerons': 3,
                          'bottom_diameter': 100.0,
                          'young_modulus': 3500.0,
                          'shear_modulus': 1287.0,
                          'd': 4.173437499999999,
                          'pitch': 120.703125,
                          'top_diameter': 83.75,
                          'imperfection': 0.052175599251571225,
                          'section': 'circular'}, }


#%% create and dump data

variables = transform_inputs_supercompressible(variables[inputs_type])
print(variables)

# create data
data = OrderedDict({'abstract_model': 'f3das.abaqus.models.supercompressible.SupercompressibleModel',
                    'variables': variables,
                    'sim_info': OrderedDict({'SUPERCOMPRESSIBLE_LIN_BUCKLE':
                                             {'sim_type': 'lin_buckle',
                                              'job_name': 'Simul_supercompressible_lin_buckle',
                                              'job_description': ''},
                                             'SUPERCOMPRESSIBLE_RIKS':
                                             {'sim_type': 'riks',
                                                 'job_name': 'Simul_supercompressible_riks',
                                                 'job_description': ''}})})

# create directory
os.mkdir(dir_name)

# create pickle
with open(os.path.join(dir_name, filename), 'wb') as file:
    pickle.dump(data, file, protocol=2)


#%% run abaqus

# create run filename
run_filename = verify_existing_name('_temp.py')
module_name = 'f3das.abaqus.run.run_model'
lines = ['import runpy',
         'import os',
         'import sys',
         'initial_wd = os.getcwd()',
         'sys.path.append(initial_wd)',
         "os.chdir(r'%s')" % dir_name,
         "runpy.run_module('%s', run_name='__main__')" % module_name,
         'os.chdir(initial_wd)']
with open(run_filename, 'w') as f:
    for line in lines:
        f.write(line + '\n')

# open abaqus and run module
interface_cmd = 'SCRIPT' if gui else 'noGUI'
command = 'abaqus cae %s=%s ' % (interface_cmd, run_filename)
os.system(command)


#%% clear direction

os.remove(run_filename)
