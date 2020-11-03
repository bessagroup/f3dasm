'''
Created on 2020-11-03 10:40:32
Last modified on 2020-11-03 10:43:15

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# third-party
import numpy as np

# local library
from .linalg import sqrtm


# function definition

def compute_small_strains_from_green(epsilon_green):
    '''
    Notes
    -----
    Assumes R=1.
    '''
    identity = np.identity(2)
    def_grad = sqrtm(2 * epsilon_green + identity)

    return 1 / 2 * (def_grad + np.transpose(def_grad)) - identity
