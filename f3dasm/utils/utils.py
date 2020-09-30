'''
Created on 2020-05-05 13:51:00
Last modified on 2020-09-25 11:22:05

@author: L. F. Pereira (lfpereira@fe.up.pt)
'''

# imports

# standard library
import gzip
import pickle
import importlib


# object definition

def get_int_number_from_str(var):
    '''
    Returns integer number from a str. e.g. 'DoE_point1' -> 1.
    '''
    return int(var[len(var.rstrip('0123456789')):])


def read_pkl_file(filename):
    '''
    Verify if file is compressed or not.
    '''

    try:
        with gzip.open(filename, 'rb') as file:
            data = pickle.load(file)
    except UnpicklingError:
        with open(filename, 'rb') as file:
            data = pickle.load(file)

    return data


def import_abstract_obj(fnc_loc):
    module_name, method_name = fnc_loc.rsplit('.', 1)
    module = importlib.import_module(module_name)

    return getattr(module, method_name)
