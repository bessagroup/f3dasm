'''
Created on 2020-10-15 09:30:17
Last modified on 2020-12-16 10:41:06

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# abaqus library
from abaqus import mdb, backwardCompatibility
from abaqusConstants import OFF

# third-party
from f3dasm.abaqus.geometry.rve import RVE2D
# from f3dasm.abaqus.geometry.shapes import Circle
from f3dasm.abaqus.geometry.shapes import PeriodicCircle as Circle
from f3dasm.abaqus.modelling.step import StaticStep
from f3dasm.abaqus.material.abaqus_materials import AbaqusMaterial
from f3dasm.abaqus.material.section import HomogeneousSolidSection
from f3dasm.abaqus.modelling.outputs import del_default_outputs
from f3dasm.abaqus.modelling.outputs import HistoryOutputRequest
from f3dasm.abaqus.modelling.outputs import FieldOutputRequest


# initialization

backwardCompatibility.setValues(reportDeprecated=False)

model_name = 'RVE2D'
job_name = 'Sim_' + model_name
job_description = ''

# bcs
eps_11 = 1.
eps_22 = -0.1
eps_12 = 0.2
epsilon = [eps_11, eps_12, eps_22]

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
material_name = 'YET_ANOTHER_STEEL'
props = {'E': 190e3,
         'nu': .3, }
fiber_material2 = AbaqusMaterial(name=material_name, props=props,
                                 section=HomogeneousSolidSection())

# rve
length = 1.
width = 1.
center = (.5, .5)
rve = RVE2D(length=length, width=width, material=matrix_material, center=center)
dims = [length, width]
bounds = [(c - dim / 2, c + dim / 2) for dim, c in zip(dims, center)]
rve.add_particle(Circle(name='PARTICLE_1', center=[0.5, 0.5], r=0.1,
                        material=None, bounds=bounds))
rve.add_particle(Circle(name='PARTICLE_2', center=[-0.1, 0.5], r=0.2,
                        material=fiber_material, bounds=bounds))
rve.add_particle(Circle(name='PARTICLE_3', center=[0.5, -0.1], r=0.2,
                        material=fiber_material2, bounds=bounds))
mesh_size = .1
rve.mesh.change_definitions(size=mesh_size, deviation_factor=0.1,
                            min_size_factor=0.1)

# create part and assembly

# create material
matrix_material.create(model)
fiber_material.create(model)
fiber_material2.create(model)

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


# set field outputs
history = HistoryOutputRequest(name='ENERGIES', createStepName=step_name,
                               variables=('ALLEN',))
field = FieldOutputRequest(name='STRESS_STRAIN_DISP', createStepName=step_name,
                           variables=('S', 'U', 'LE'))

history.create(model)
field.create(model)
del_default_outputs(model)


# create inp
modelJob = mdb.Job(model=model_name, name=job_name)
modelJob.writeInput(consistencyChecking=OFF)
modelJob.submit()
