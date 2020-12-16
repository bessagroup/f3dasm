'''
Created on 2020-04-08 10:27:22
Last modified on 2020-11-02 12:03:08

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Store material data.
'''


# imports

# local library
from .utils import Property


# create materials

t800_17GSM_120 = {
    'E1': 1.28e+11,
    'E2': 6.5e+09,
    'nu12': 0.35,
    'G12': 7.5e+09,
    'G13': 7.5e+09,
    'G23': 7.5e+09,
    'rho': 1.0,
}

steel_elastic = {
    'E': 210e9,
    'nu': .32,
    'rho': 7.8e3,
}
