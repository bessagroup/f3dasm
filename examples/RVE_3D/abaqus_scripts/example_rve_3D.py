'''
Created on 2020-10-15 09:30:17
Last modified on 2020-11-24 13:50:39

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# abaqus library
from abaqus import mdb, backwardCompatibility

# third-party
from f3dasm.abaqus.geometry.rve import RVE3D
# from f3dasm.abaqus.geometry.shapes import Sphere
from f3dasm.abaqus.geometry.shapes import PeriodicSphere as Sphere
from f3dasm.abaqus.material.abaqus_materials import AbaqusMaterial
from f3dasm.abaqus.material.section import HomogeneousSolidSection


# initialization

# backwardCompatibility.setValues(reportDeprecated=False)

model_name = 'RVE3D'
job_name = 'Sim_' + model_name
job_description = ''


# create model

model = mdb.Model(name=model_name)

if 'Model-1' in mdb.models.keys():
    del mdb.models['Model-1']

# define objects

# define material
material_name = 'STEEL'
props = {'E': 210e3,
         'nu': .3, }
matrix_material = AbaqusMaterial(name=material_name, props=props,
                                 section=HomogeneousSolidSection())
material_name = 'OTHER_STEEL'
props = {'E': 200e3,
         'nu': .3, }
fiber_material = AbaqusMaterial(name=material_name, props=props,
                                section=HomogeneousSolidSection())

# rve
dims = (1., 1., 1.)
center = (.5, .5, .5)
rve = RVE3D(dims=dims, material=matrix_material, center=center)
bounds = [(c - dim / 2, c + dim / 2) for dim, c in zip(dims, center)]
rve.add_particle(Sphere(name='PARTICLE_1', center=[1.1, 1.2, 1.], r=0.25,
                        material=fiber_material, bounds=bounds))
# rve.add_particle(Sphere(name='PARTICLE_2', center=[0., 0., 0.], r=0.25,
#                         material=None))
mesh_size = .1
rve.mesh.change_definitions(size=mesh_size, deviation_factor=0.1,
                            min_size_factor=0.1)

# create part and assembly

# create material
matrix_material.create(model)
fiber_material.create(model)

rve.create_part(model)
rve.create_instance(model)
# success = rve.generate_mesh(simple_trial=True, face_by_closest=False)
# print('Mesh generated successfully? {}'.format(success))

# # apply boundary conditions
# rve.apply_pbcs_constraints(model)
