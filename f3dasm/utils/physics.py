'''
Created on 2020-04-26 00:38:18
Last modified on 2020-04-26 00:45:37
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Functions to compute geometric/physical properties.
'''


#%% imports

# third-party
import numpy as np


#%% function definition

def get_circular_section_props(d):
    Ixx = Iyy = np.pi * d**4 / 64
    J = Ixx * 2
    A = np.pi * d**2 / 4

    return Ixx, Iyy, J, A
