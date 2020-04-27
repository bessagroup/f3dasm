'''
Created on 2020-04-26 04:41:27
Last modified on 2020-04-26 05:50:59
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Run 3d DoE.
'''


#%% imports

# standard library
import os

# local library
from f3das.misc.file_handling import verify_existing_name


#%% initialization

# folder structure
example_name = 'example_supercompressible'
simuls_dir_name = 'analyses'

# points to run
points = list(range(2500, 3000)) + list(range(6500, 7000))


#%% run abaqus

# create run filename
run_filename = verify_existing_name('_temp.py')
module_name = 'f3das.abaqus.run.run_model'
lines = ['import runpy',
         'import os',
         'import sys',
         'initial_wd = os.getcwd()',
         'sys.path.append(initial_wd)',
         'points = %s' % points,
         "sim_dir = r'%s'" % os.path.join(example_name, simuls_dir_name),
         'for point in points:',
         "\tos.chdir('%s' % os.path.join(sim_dir, 'DoE_point%i' % point))",
         "\trunpy.run_module('%s', run_name='__main__')" % module_name,
         '\tos.chdir(initial_wd)']
with open(run_filename, 'w') as f:
    for line in lines:
        f.write(line + '\n')

# open abaqus and run module
command = 'abaqus cae noGUI=%s ' % run_filename
os.system(command)


#%% clear direction

os.remove(run_filename)
