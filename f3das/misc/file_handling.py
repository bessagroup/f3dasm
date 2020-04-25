'''
Created on 2020-04-25 15:56:27
Last modified on 2020-04-25 17:39:36
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Define functions used to manipulate files.
'''


#%% imports

# standard library
import os

# third-party
import numpy as np


#%% function definition

def verify_existing_name(name_init):
    try:
        name, ext = os.path.splitext(name_init)
    except ValueError:
        name = name_init
        ext = ''
    filename = name_init
    i = 1
    while os.path.exists(filename):
        i += 1
        filename = name + '(%s)' % str(i) + ext

    return filename


def get_sorted_by_time(parcial_name, dir_path=None, ext=None):
    '''
    Gets the most recent created folder that contains part of a given name.
    '''

    # initialization
    dir_path = os.getcwd() if dir_path is None else dir_path
    potential_file_names = []
    created_times = []

    # get created times
    if ext is None:
        potential_file_names = [name for name in os.listdir(dir_path)
                                if os.path.isdir(name) and parcial_name in name]
    else:
        potential_file_names = [name for name in os.listdir(dir_path)
                                if name.endswith('.' + ext) and parcial_name in name]
    created_times = [os.path.getctime(os.path.join(dir_path, name))
                     for name in potential_file_names]
    indices = np.argsort(created_times)

    # find most recent folder
    filenames = [potential_file_names[index] for index in indices]

    return filenames
