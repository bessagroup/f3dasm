'''
Created on 2020-04-08 10:27:22
Last modified on 2020-09-11 17:06:44
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Store material data.

Notes
-----
-Possibly, this is not the most logical way to store material data.
Nevertheless, after the user gets used to this strategy, its advantages will
become clear. It is possible better than the use of databases, because they are
less flexible and do not deal well with tabular data (which is common due
data dependency on e.g. temperature).
-Materials are stored in dictionaries.
-Each key in the dictionary has to be in accordance with the respective
material class, i.e. all the necessary variables have to be provided with
the corresponding name.
-Each field in dictionary contains an instance of the class Property or a float
(in that case, it is assumed that the float corresponds to the variable value).
-The class Property is prepared to receive both scalar and tabular data, as
well as information about the property deviation, unit and additional info.
Whenever a parameter is not applicable (e.g. property deviation), None must be
used.
'''


#%% imports

# local library
from .utils import Property


#%% create materials


t800_17GSM_120 = {
    'E1': 1.28e+11,
    'E2': 6.5e+09,
    'nu12': 0.35,
    'G12': 7.5e+09,
    'G13': 7.5e+09,
    'G23': 7.5e+09,
    'density': 1.0,
}

steel_elastic = {
    'E': 210e9,
    'nu': .32,
    'density': 7.8e3,
}
