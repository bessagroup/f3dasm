'''
Created on 2020-03-24 14:52:25
Last modified on 2020-11-17 15:15:31

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Show how to generate a 2d RVE (in particular, Bertoldi's RVE).
'''


# imports

# abaqus library
from abaqus import mdb, backwardCompatibility
from abaqusConstants import OFF

# third-party
from f3dasm.abaqus.geometry.rve import RVE2D
from f3dasm.abaqus.modelling.step import StaticStep

# local library
from examples.Bertoldi.abaqus_modules.bertoldi_rve import BertoldiPore


# initialization

backwardCompatibility.setValues(reportDeprecated=False)

model_name = 'RVE2D'
job_name = 'Sim_' + model_name
job_description = ''

# geometry
length = 4.5
width = 3.5
center = (1.75, 1.75)

# inner shape
r_0 = 1.
c_1 = 2.98642006e-01
c_2 = 1.37136137e-01

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

rve = RVE2D(length, width, center, name='BERTOLDI_RVE')
rve.mesh.change_definitions(size=0.1)

pore = BertoldiPore(center, r_0, c_1, c_2)
rve.add_particle(pore)


# create part and assembly

# create part and generate mesh
rve.create_part(model)
success = rve.generate_mesh()
print('Mesh generated successfully? {}'.format(success))

# create assembly
rve.create_instance(model)

# set step
step_name = "STATIC_STEP"
static_step = StaticStep(step_name, initialInc=0.02, timePeriod=1.,
                         minInc=1e-5, maxInc=0.02)
static_step.create(model)

# # TODO: add initial conditions

# set boundary conditions
constraints, bcs = rve.bcs.set_bcs(step_name, epsilon,
                                   green_lagrange_strain=False)

constraints.create(model)
for bc in bcs:
    bc.create(model)


# # create inp
# # modelJob = mdb.Job(model=model_name, name=job_name)
# # modelJob.writeInput(consistencyChecking=OFF)
