'''
Created on 2020-04-29 11:02:24
Last modified on 2020-05-09 21:30:21
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------

'''

#%% imports

# standard library
import os
import pickle


#%%  initialization

# folder structure
example_name = 'example_supercompressible_3d_circular'
simuls_dir_name = 'analyses'
cwd = os.getcwd()

folder_names = os.listdir(os.path.join(example_name, simuls_dir_name))

for folder in folder_names:
    folder_dir = os.path.join(cwd, example_name, simuls_dir_name, folder)
    os.chdir(folder_dir)
    if os.path.isdir(folder_dir):

        try:
            filename = 'simul.pkl_abaqus'
            with open(filename, 'rb') as file:
                data = pickle.load(file)

            try:
                del data['post-processing']
            except:
                pass
            try:
                del data['time']
            except:
                pass

            with open(filename, 'wb') as file:
                pickle.dump(data, file, protocol=2)
        except:
            pass
