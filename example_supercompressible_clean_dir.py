'''
Created on 2020-04-29 11:02:24
Last modified on 2020-04-29 11:07:37
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------

'''

#%% imports

# standard library
import os


#%%  initialization

# folder structure
example_name = 'example_supercompressible_7d'
simuls_dir_name = 'analyses'


for folder in os.listdir(os.path.join(example_name, simuls_dir_name)):
    folder_dir = os.path.join(example_name, simuls_dir_name, folder)
    if os.path.isdir(folder_dir):
        for filename in os.listdir(folder_dir):
            if not filename.endswith('.pkl') and not filename.endswith('.pkl_abaqus'):
                os.remove(os.path.join(folder_dir, filename))
