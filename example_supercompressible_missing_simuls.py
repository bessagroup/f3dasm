'''
Created on 2020-04-28 16:52:22
Last modified on 2020-04-28 17:07:58
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)
'''


#%% imports

# local library
from f3das.abaqus.run.misc import get_missing_simuls


#%% initialization

example_name = 'example_supercompressible'
simuls_dir_name = 'analyses'


#%% get missing simuls

missing_simuls = get_missing_simuls(example_name, simuls_dir_name)

print(missing_simuls)
print(len(missing_simuls))
