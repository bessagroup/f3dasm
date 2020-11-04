'''
Created on 2020-04-08 13:57:51
Last modified on 2020-11-04 13:12:14

@author: L. F. Pereira (lfpereira@fe.up.pt)
'''


# imports

# abaqus
from abaqus import mdb, backwardCompatibility
from abaqusConstants import ON


# third-party
from f3dasm.abaqus.material.abaqus_materials import AbaqusMaterial
from f3dasm.abaqus.material.abaqus_materials import ElasticIsotropicBehavior
from f3dasm.abaqus.material.abaqus_materials import ElasticLaminaBehavior
from f3dasm.abaqus.material.abaqus_materials import ElasticEngineeringConstantsBehavior
from f3dasm.abaqus.material.abaqus_materials import Density
from f3dasm.abaqus.material.abaqus_materials import ExpansionIsotropicBehavior


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
props_iso_elastic = {
    'E': 210e9,
    'nu': .32,
    'rho': 7.8e3,
    'alpha': 4.9E-6,
    'T0': 120,
}

# without specifying behaviors
iso_1 = AbaqusMaterial(
    name='isotropic_no_beh', props=props_iso_elastic,
    model=model, create_section=True)

# specifying behavior (properties given to behavior)
iso_beh = ElasticIsotropicBehavior(props=props_iso_elastic, noCompression=ON)
den_beh = Density(props=props_iso_elastic)
exp_beh = ExpansionIsotropicBehavior(props=props_iso_elastic)
iso_2 = AbaqusMaterial(
    name='isotropic_beh_no_props',
    material_behaviors=[iso_beh, den_beh, exp_beh], model=model,
    create_section=True)

# specifying behavior (properties given to material)
iso_beh = ElasticIsotropicBehavior()
den_beh = Density()
exp_beh = ExpansionIsotropicBehavior()
iso_3 = AbaqusMaterial(
    name='isotropic_beh_props', props=props_iso_elastic,
    material_behaviors=[iso_beh, den_beh, exp_beh], model=model,
    create_section=True)


# isotropic with temperature dependency
props_iso_elastic_temp = {
    'T': [10., 20., 30.],
    'E': [210e9, 200e9, 190e9],
    'nu': [.32, .32, 0.32],
    'rho': 7.8e3,
    'alpha': 4.9E-6,
    'T0': 120,
}
# without specifying behaviors
iso_temp_1 = AbaqusMaterial(
    name='isotropic_temp_no_beh', props=props_iso_elastic_temp,
    model=model, create_section=True)

# specifying behavior (properties given to behavior)
iso_beh = ElasticIsotropicBehavior(props=props_iso_elastic_temp)
den_beh = Density(props=props_iso_elastic_temp)
exp_beh = ExpansionIsotropicBehavior(props=props_iso_elastic_temp)
iso_temp_2 = AbaqusMaterial(
    name='isotropic_temp_beh_no_props',
    material_behaviors=[iso_beh, den_beh, exp_beh], model=model,
    create_section=True)

# specifying behavior (properties given to material)
iso_beh = ElasticIsotropicBehavior()
den_beh = Density()
exp_beh = ExpansionIsotropicBehavior()
iso_temp_3 = AbaqusMaterial(
    name='isotropic_temp_beh_props', props=props_iso_elastic_temp,
    material_behaviors=[iso_beh, den_beh, exp_beh], model=model,
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
eng_const_beh = ElasticEngineeringConstantsBehavior(props=props_eng_const)
den_beh = Density(props=props_eng_const)
eng_const_2 = AbaqusMaterial(
    name='eng_const_beh_no_props',
    material_behaviors=[eng_const_beh, den_beh], model=model,
    create_section=True)

# specifying behavior (properties given to material)
eng_const_beh = ElasticEngineeringConstantsBehavior()
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
lam_1 = AbaqusMaterial('lamina_no_beh', props=props_lamina,
                       model=model, create_section=False)

# specifying behavior (properties given to behavior)
lam_beh = ElasticLaminaBehavior(props=props_lamina)
den_beh = Density(props=props_lamina)
lam_2 = AbaqusMaterial(
    name='lamina_beh_no_props',
    material_behaviors=[lam_beh, den_beh], model=model,
    create_section=True)

# specifying behavior (properties given to material)
lam_beh = ElasticLaminaBehavior()
den_beh = Density()
lam_3 = AbaqusMaterial(
    name='lamina_beh_props', props=props_lamina,
    material_behaviors=[lam_beh, den_beh], model=model,
    create_section=True)
