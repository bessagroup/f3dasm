'''
Created on 2020-05-05 13:51:00
Last modified on 2020-05-05 13:53:31
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Define additional functions.
'''


def get_int_number_from_str(var):
    '''
    Returns integer number from a str. e.g. 'DoE_point1' -> 1.
    '''
    return int(var[len(var.rstrip('0123456789')):])
