'''
Created on 2020-10-15 09:30:17
Last modified on 2020-12-01 16:16:17

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# abaqus library
from abaqus import mdb, backwardCompatibility
from abaqusConstants import OFF

# third-party
from f3dasm.abaqus.geometry.rve import RVE3D
# from f3dasm.abaqus.geometry.shapes import Sphere
from f3dasm.abaqus.geometry.shapes import PeriodicSphere as Sphere
from f3dasm.abaqus.modelling.step import StaticStep
from f3dasm.abaqus.material.abaqus_materials import AbaqusMaterial
from f3dasm.abaqus.material.section import HomogeneousSolidSection


# initialization

backwardCompatibility.setValues(reportDeprecated=False)

model_name = 'RVE3D'
job_name = 'Sim_' + model_name
job_description = ''

# bcs
eps_11 = 1.
eps_22 = -0.1
eps_33 = 1.2
eps_12 = 0.2
eps_13 = 0.4
eps_23 = .5
epsilon = [eps_11, eps_12, eps_13, eps_22, eps_23, eps_33]


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
rve = RVE3D(dims=dims, material=matrix_material, center=center,
            mesh_strat='S1', constrain_strat='by_sorting')
bounds = [(c - dim / 2, c + dim / 2) for dim, c in zip(dims, center)]
rve.add_particle(Sphere(name='PARTICLE_1', center=center, r=0.25,
                        material=fiber_material, bounds=bounds))
rve.add_particle(Sphere(name='PARTICLE_2', center=[0., 0., 0.], r=0.25,
                        material=fiber_material, bounds=bounds))
mesh_size = .1
rve.mesh.change_definitions(size=mesh_size, deviation_factor=0.1,
                            min_size_factor=0.1)

# create part and assembly

# create material
matrix_material.create(model)
fiber_material.create(model)

rve.create_part(model)
rve.create_instance(model)
success = rve.generate_mesh()
print('Mesh generated successfully? {}'.format(success))

# set step
step_name = "STATIC_STEP"
static_step = StaticStep(step_name, initialInc=0.02, timePeriod=1.,
                         minInc=1e-5, maxInc=0.02)
static_step.create(model)

# set boundary conditions
constraints, bcs = rve.bcs.set_bcs(step_name, epsilon,
                                   green_lagrange_strain=False)

constraints.create(model)
for bc in bcs:
    bc.create(model)


# create inp
modelJob = mdb.Job(model=model_name, name=job_name)
modelJob.writeInput(consistencyChecking=OFF)
modelJob.submit()
