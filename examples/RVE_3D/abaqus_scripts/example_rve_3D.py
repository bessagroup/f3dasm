'''
Created on 2020-10-15 09:30:17
Last modified on 2020-11-12 12:35:56

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# abaqus library
from abaqus import mdb, backwardCompatibility

# third-party
from f3dasm.abaqus.geometry.rve import RVE3D
from f3dasm.abaqus.geometry.shapes import Sphere


# local functiosn
def get_RVE_info_from_file(filename):
    with open(filename, 'r') as file:
        data = file.readlines()

    # create rve
    dims = [float(dim) for dim in data[0].split()]

    # add particles
    centers, radii = [], []
    for line in data[1:]:
        particle_info = line.split()
        centers.append([float(pos) for pos in particle_info[:-1]])
        radii.append(float(particle_info[-1]))

    return dims, centers, radii


# initialization

backwardCompatibility.setValues(reportDeprecated=False)

model_name = 'RVE3D'
job_name = 'Sim_' + model_name
job_description = ''

# initialization
# filename = 'particles.txt'
# periodic = True
# filename = 'particles_all.txt'
# periodic = False
filename = None

# create model

model = mdb.Model(name=model_name)

if 'Model-1' in mdb.models.keys():
    del mdb.models['Model-1']

# define objects
if filename:
    dims, centers, radii = get_RVE_info_from_file(filename)
    rve = RVE3D(dims)
    for i, (center, radius) in enumerate(zip(centers, radii)):
        rve.add_particle(Sphere(name='PARTICLE_{}'.format(i), center=center,
                                r=radius, periodic=periodic, dims=rve.dims))
else:
    rve = RVE3D(dims=[1., 1., 1.], center=(.5, .5, .5))
    rve.add_particle(Sphere(name='PARTICLE_1', center=[.5, .5, .5], r=0.25))
    # rve.add_particle(Sphere(name='PARTICLE_2', center=[1.1, 1.2, 1.], r=0.25,
    #                         periodic=True, dims=rve.dims))
    radii = [0.5]  # fake for code not fail

# mesh_size = min(radii) / 5.
mesh_size = .1
print('mesh_size: {:.4f}'.format(mesh_size))
rve.change_mesh_definitions(mesh_size=mesh_size)
# rve.change_mesh_definitions(mesh_deviation_factor=0.1, mesh_min_size_factor=0.1)

# create part and assembly
rve.create_part(model)
rve.create_instance(model)
rve.generate_mesh(simple_trial=True, face_by_closest=False)

# apply boundary conditions
rve.apply_pbcs_constraints(model)
