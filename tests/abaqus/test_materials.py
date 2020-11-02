'''
Created on 2020-04-08 13:57:51
Last modified on 2020-11-02 12:01:06

@author: L. F. Pereira (lfpereira@fe.up.pt)
'''


# imports

# abaqus
from abaqus import mdb, backwardCompatibility


# third-party
from f3dasm.abaqus.material.abaqus_materials import AbaqusMaterial
from f3dasm.abaqus.material.abaqus_materials import IsotropicBehavior
from f3dasm.abaqus.material.abaqus_materials import LaminaBehavior
from f3dasm.abaqus.material.abaqus_materials import EngineeringConstantsBehavior
from f3dasm.abaqus.material.abaqus_materials import Density


# initialization

backwardCompatibility.setValues(reportDeprecated=False)

model_name = 'TEST-MATERIALS'
job_name = 'Sim_' + model_name
job_description = ''


# create model

model = mdb.Model(name=model_name)

if 'Model-1' in mdb.models.keys():
    del mdb.models['Model-1']


# define and create materials

# isotropic
props_isotropic_elastic = {
    'E': 210e9,
    'nu': .32,
    'rho': 7.8e3,
}

# without specifying behaviors
isotropic_1 = AbaqusMaterial(
    name='isotropic_no_beh', props=props_isotropic_elastic,
    model=model, create_section=True)

# specifying behavior (properties given to behavior)
iso_beh = IsotropicBehavior(props=props_isotropic_elastic)
den_beh = Density(props=props_isotropic_elastic)
isotropic_2 = AbaqusMaterial(
    name='isotropic_beh_no_props',
    material_behaviors=[iso_beh, den_beh], model=model,
    create_section=True)

# specifying behavior (properties given to material)
iso_beh = IsotropicBehavior()
den_beh = Density()
isotropic_3 = AbaqusMaterial(
    name='isotropic_beh_props', props=props_isotropic_elastic,
    material_behaviors=[iso_beh, den_beh], model=model,
    create_section=True)


# eng_consttropic
props_eng_const = {
    'E1': 1.28e+11,
    'E2': 6.5e+09,
    'E3': 6.5e+09,
    'nu12': 0.35,
    'nu13': 0.1,
    'nu23': 0.1,
    'G12': 7.5e+09,
    'G13': 7.5e+09,
    'G23': 7.5e+09,
    'rho': 1.0,
}

# without specifying behavior
eng_const_1 = AbaqusMaterial('eng_const_no_beh', props=props_eng_const,
                             model=model, create_section=False)

# specifying behavior (properties given to behavior)
eng_const_beh = EngineeringConstantsBehavior(props=props_eng_const)
den_beh = Density(props=props_eng_const)
eng_const_2 = AbaqusMaterial(
    name='eng_const_beh_no_props',
    material_behaviors=[eng_const_beh, den_beh], model=model,
    create_section=True)

# specifying behavior (properties given to material)
eng_const_beh = EngineeringConstantsBehavior()
den_beh = Density()
eng_const_3 = AbaqusMaterial(
    name='eng_const_beh_props', props=props_eng_const,
    material_behaviors=[eng_const_beh, den_beh], model=model,
    create_section=True)


# lamina
props_lamina = {
    'E1': 1.28e+11,
    'E2': 6.5e+09,
    'nu12': 0.35,
    'G12': 7.5e+09,
    'G13': 7.5e+09,
    'G23': 7.5e+09,
    'rho': 1.0,
}

# without specifying behavior
lamina_1 = AbaqusMaterial('lamina_no_beh', props=props_lamina,
                          model=model, create_section=False)

# specifying behavior (properties given to behavior)
lam_beh = LaminaBehavior(props=props_lamina)
den_beh = Density(props=props_lamina)
lamina_2 = AbaqusMaterial(
    name='lamina_beh_no_props',
    material_behaviors=[lam_beh, den_beh], model=model,
    create_section=True)

# specifying behavior (properties given to material)
lam_beh = LaminaBehavior()
den_beh = Density()
lamina_3 = AbaqusMaterial(
    name='lamina_beh_props', props=props_lamina,
    material_behaviors=[lam_beh, den_beh], model=model,
    create_section=True)
