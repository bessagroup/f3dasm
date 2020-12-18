'''
Created on 2020-12-18 17:19:03
Last modified on 2020-12-18 18:11:23

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''


# imports

# abaqus
from abaqus import backwardCompatibility

# local library
from modules.rve_2d_circles import make_rve_2d_circles


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

make_rve_2d_circles(model_name, job_name, job_description, epsilon)
