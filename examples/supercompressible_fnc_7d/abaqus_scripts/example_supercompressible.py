'''
Created on 2020-09-21 13:50:32
Last modified on 2020-09-30 14:27:45

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''


# imports

# abaqus
from abaqus import mdb
from abaqusConstants import OFF
from abaqus import session

# third-party
from f3dasm.abaqus.models.supercompressible_fnc import lin_buckle
from f3dasm.abaqus.models.supercompressible_fnc import post_process_lin_buckle
from f3dasm.abaqus.models.supercompressible_fnc import riks
from f3dasm.abaqus.models.supercompressible_fnc import post_process_riks


# initialization
model_name = 'SUPERCOMPRESSIBLE'


n_longerons = 3
bottom_diameter = 100.
top_diameter = 82.42
pitch = 1.15223e2
young_modulus = 3.5e3
shear_modulus = 1.38631e3
cross_section_props = {'area': 1.00004e+01,
                       'Ixx': 5.24157e+01,
                       'Iyy': 7.50000e+01,
                       'J': 2.50000e+02}
imperfection = 7.85114e-02


# create, run and post-process linear buckling

sim_type = 'lin_buckle'
job_info = {'name': 'Simul_{}_{}'.format(model_name, sim_type)}
lin_buckle(model_name, job_info['name'], n_longerons, bottom_diameter,
           top_diameter, pitch, young_modulus, shear_modulus,
           cross_section_props,)

modelJob = mdb.JobFromInputFile(inputFileName='{}.inp'.format(job_info['name']),
                                **job_info)
modelJob.submit(consistencyChecking=OFF)
modelJob.waitForCompletion()

odb_name = '%s.odb' % job_info['name']
odb = session.openOdb(name=odb_name)
data_lin_buckle = post_process_lin_buckle(odb)
print('\nLinear buckling results')
print(data_lin_buckle)


# create, run and post-process riks

sim_type = 'riks'
job_info = {'name': 'Simul_{}_{}'.format(model_name, sim_type)}
riks(job_name=job_info['name'], imperfection=imperfection,
     previous_model_results=data_lin_buckle)

modelJob = mdb.JobFromInputFile(inputFileName='{}.inp'.format(job_info['name']),
                                **job_info)
modelJob.submit(consistencyChecking=OFF)
modelJob.waitForCompletion()

odb_name = '%s.odb' % job_info['name']
odb = session.openOdb(name=odb_name)
data_riks = post_process_riks(odb)
print('\nRiks results')
print(data_riks)
