'''
Created on 2019-09-12 16:51:02
Last modified on 2020-05-07 16:06:20
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Run Abaqus examples that are presented in examples/abaqus.

Notes
-----
-to run in the cmd: >>> abaqus python run_abaqus_example.py <example_name> --gui
-if --gui is not added to cmd, then it runs without gui
-Abaqus work directory will be the one that contains this file (this file needs
to get access to f3das).
'''


#%% imports

# standard library
import os
import argparse
from itertools import chain

# local library
from f3das.misc.file_handling import verify_existing_name
from f3das.misc.file_handling import clean_abaqus_dir


#%% parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('example_name', type=str,
                    help='Example name. It should be in the folder Abaqus examples')
parser.add_argument('--gui',
                    action='store_true',
                    help='Abaqus cae interface is opened')

args = parser.parse_args()


#%% find example in subfolders

folder_names = ['examples']
example_name = os.path.splitext(args.example_name)[0] + '.py'

for path, _, filenames in chain.from_iterable([os.walk(folder_name) for folder_name in folder_names]):
    for filename in filenames:
        if filename.endswith(example_name):
            name = os.path.splitext(os.path.join(path, filename))[0]
            break


#%%  generate run file

run_filename = verify_existing_name('run.py')
module_name = '.'.join(os.path.normpath(name).split(os.sep))
lines = ['import runpy',
         "runpy.run_module('%s', run_name='__main__')" % module_name]
with open(run_filename, 'w') as f:
    for line in lines:
        f.write(line + '\n')


#%% generate example

interface_cmd = 'SCRIPT' if args.gui else 'noGUI'
command = 'abaqus cae ' + interface_cmd + '=%s' % run_filename
fail = os.system(command)


#%% clean abaqus dir and finalize script

clean_abaqus_dir()
os.remove(run_filename)

if fail:
    raise Exception("'%s' has a bug!" % os.path.splitext(example_name)[0])
else:
    print("'%s' successfully run" % os.path.splitext(example_name)[0])
