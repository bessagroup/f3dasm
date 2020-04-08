'''
Created on 2020-04-08 13:57:51
Last modified on 2020-04-08 14:16:51
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

# local library
from src.abq.material.abaqus_materials import IsotropicMaterial
from src.abq.material.abaqus_materials import LaminaMaterial


#%% initialization

backwardCompatibility.setValues(reportDeprecated=False)

model_name = 'TEST-MATERIALS'
job_name = 'Sim_' + model_name
job_description = ''

# define materials
isotropic_material_name = 'steel_elastic'
lamina_material_name = 't800_17GSM_120'


#%% create model

model = mdb.Model(name=model_name)

if 'Model-1' in mdb.models.keys():
    del mdb.models['Model-1']


#%% create materials

isotropic_material = IsotropicMaterial(isotropic_material_name, model=model,
                                       create_section=True)
lamina_material = LaminaMaterial(lamina_material_name, model=model,
                                 create_section=False)
