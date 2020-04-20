'''
Created on 2020-04-08 11:56:46
Last modified on 2020-04-20 22:53:20
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Verify that material subpackage is working properly.
'''


#%% imports

# standard library
import pickle

# local library
from src.material.material import Material


#%% initialization

material_name = 't800_17GSM_120'


#%% create material and print properties

material = Material(material_name, read=True)
property_values = {key: value.value for key, value in material.props.items()}

print(property_values)


#%% dump object

data = {'material': material}
filename = 'test.pickle'
with open(filename, 'wb') as f:
    pickle.dump(data, f)

# with open(filename, 'rb') as f:
#     data = pickle.load(f)