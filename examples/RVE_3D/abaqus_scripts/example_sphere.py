'''
Created on 2020-10-15 09:30:17
Last modified on 2020-10-15 17:26:36

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# abaqus library
from abaqus import mdb, backwardCompatibility

# third-party
from f3dasm.abaqus.geometry.rve import RVE3D
from f3dasm.abaqus.geometry.shapes import Sphere


# initialization

backwardCompatibility.setValues(reportDeprecated=False)

model_name = 'RVE3D'
job_name = 'Sim_' + model_name
job_description = ''

# geometry
dims = [1., 1., 1.]


# create model

model = mdb.Model(name=model_name)

if 'Model-1' in mdb.models.keys():
    del mdb.models['Model-1']


# define objects
rve = RVE3D(dims)
sphere = Sphere(name='PARTICLE_1', center=(1., 1., 0.5), r=0.25)
rve.add_particle(sphere)

# create part and assembly
sphere.create_part(model)
sphere.create_instance(model)
sphere.generate_mesh()
