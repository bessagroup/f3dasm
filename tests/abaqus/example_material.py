'''
Created on 2020-04-08 13:57:51
Last modified on 2020-09-30 11:36:33
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Show how to create material using the classes defined in abq.material
'''


#%% imports

# abaqus
from abaqus import mdb, backwardCompatibility

# standard library
import pickle

# third-party
from f3dasm.abaqus.material.abaqus_materials import IsotropicMaterial
from f3dasm.abaqus.material.abaqus_materials import LaminaMaterial


# initialization

backwardCompatibility.setValues(reportDeprecated=False)

model_name = 'TEST-MATERIALS'
job_name = 'Sim_' + model_name
job_description = ''

# define materials
isotropic_material_name = 'steel_elastic'
lamina_material_name = 't800_17GSM_120'


# create model

model = mdb.Model(name=model_name)

if 'Model-1' in mdb.models.keys():
    del mdb.models['Model-1']


# create materials

isotropic_material = IsotropicMaterial(isotropic_material_name, model=model,
                                       create_section=True)
lamina_material = LaminaMaterial(lamina_material_name, model=model,
                                 create_section=False)


# dump object

data = {'material': lamina_material}
filename = 'test.pickle'
with open(filename, 'wb') as f:
    pickle.dump(data, f)

# with open(filename, 'rb') as f:
#     data = pickle.load(f)
