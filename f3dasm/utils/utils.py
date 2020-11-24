'''
Created on 2020-05-05 13:51:00
Last modified on 2020-11-24 11:27:08

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


def unnest(array):
    unnested_array = []
    for arrays in array:
        for array_ in arrays:
            unnested_array.append(array_)

    return unnested_array


def get_decimal_places(tol):
    d = 0
    aux = 1
    while aux > tol:
        d += 1
        aux = 10**(-d)

    return d
