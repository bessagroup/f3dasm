'''
Created on 2020-05-05 16:28:59
Last modified on 2020-05-05 16:30:17
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

# local library
from f3das.misc.file_handling import get_unique_file_by_ext


#%% function definition

def get_results(dir_name, folder_name):

    # get filename and read data
    dir_name = os.path.join(dir_name, folder_name)
    filename = get_unique_file_by_ext(dir_name, ext='.pkl')

    with open(os.path.join(dir_name, filename), 'rb') as file:
        data = pickle.load(file, encoding='latin1')

    return data
