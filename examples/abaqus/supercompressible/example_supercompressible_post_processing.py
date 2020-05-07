'''
Created on 2020-04-22 11:27:41
Last modified on 2020-04-22 12:45:42
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Show how to do post-processing in supercompressible model.

Notes
-----
-Assumes analysis was already performed and an odb exists in the abaqus
directory.
'''


#%% imports

# abaqus
from abaqus import session

# standard library
import pickle


#%% initialization

model_name = 'Simul_SUPERCOMPRESSIBLE_lin_buckle.pickle'
# model_name = 'Simul_SUPERCOMPRESSIBLE_riks.pickle'


#%% access model and perform post-processing

# read pickle
with open(model_name, 'rb') as f:
    data = pickle.load(f)
model = data['model']

# access odb
odb_name = '%s.odb' % model.job_name
odb = session.openOdb(name=odb_name)

# get and print post-processing data
data = model.perform_post_processing(odb)
print(data)
