'''
Created on 2020-04-22 14:43:42
Last modified on 2020-04-25 19:32:14
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


#%% initialization

gui = False
dir_name = os.path.join(os.getcwd(), verify_existing_name('test'))
filename = 'simul.pkl'

# geometry
n_vertices_polygon = 3
mast_diameter = 100.
mast_pitch = 115.223
cone_slope = 1.75806e-01
young_modulus = 3.50000e+03
shear_modulus = 1.38631e+03
Ixx = 6.12244e+01
Iyy = 1.26357e+01
J = 2.10974e+02
area = 1.54038e+01
mode_amplitude = 7.85114e-02


#%% create and dump data

# create data
data = OrderedDict({'abstract_model': 'f3das.abaqus.models.supercompressible.SupercompressibleModel',
                    'data': {'n_vertices_polygon': n_vertices_polygon,
                             'mast_diameter': mast_diameter,
                             'mast_pitch': mast_pitch,
                             'cone_slope': cone_slope,
                             'young_modulus': young_modulus,
                             'shear_modulus': shear_modulus,
                             'Ixx': Ixx,
                             'Iyy': Iyy,
                             'J': J,
                             'area': area,
                             'mode_amplitude': mode_amplitude, },
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
run_filename = verify_existing_name('run.py')
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
