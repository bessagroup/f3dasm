'''
Created on 2020-04-08 11:56:46
Last modified on 2020-04-08 12:02:38
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Verify that material subpackage is working properly.
'''


#%% imports

# local library
from src.material.material import Material


#%% initialization

material_name = 't800_17GSM_120'


#%% create material and print properties

material = Material(material_name, read=True)
property_values = {key: value.value for key, value in material.props.items()}

print(property_values)
