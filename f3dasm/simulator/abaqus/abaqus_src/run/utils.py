'''
Created on 2020-09-25 14:29:14
Last modified on 2020-09-25 14:31:55

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# standard library
from collections import OrderedDict


# object definition

def convert_dict_unicode_str(pickled_dict):
    new_dict = OrderedDict() if type(pickled_dict) is OrderedDict else {}
    for key, value in pickled_dict.items():
        value = _set_converter_flow(value)
        new_dict[str(key)] = value

    return new_dict


def convert_iterable_unicode_str(iterable):
    new_iterable = []
    for value in iterable:
        value = _set_converter_flow(value)
        new_iterable.append(value)

    if type(iterable) is tuple:
        new_iterable = tuple(new_iterable)
    elif type(iterable) is set:
        new_iterable = set(new_iterable)

    return new_iterable


def _set_converter_flow(value):

    if type(value) is unicode:
        value = str(value)
    elif type(value) in [OrderedDict, dict]:
        value = convert_dict_unicode_str(value)
    elif type(value) in [list, tuple, set]:
        value = convert_iterable_unicode_str(value)

    return value
