'''
Created on 2020-04-25 16:16:45
Last modified on 2020-04-25 18:47:33
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Show how the code to access results work.
'''


#%% imports

# standard library
import os
import pickle

# third-party
import numpy as np
from matplotlib import pyplot as plt

# local library
from f3das.misc.file_handling import get_sorted_by_time


#%% initialization

dir_name = os.path.join(os.getcwd(), get_sorted_by_time('test')[-1])
filename = 'simul.pkl'


#%% access data

# get data
with open(os.path.join(dir_name, filename), 'rb') as file:
    data = pickle.load(file, encoding='latin1')
sim_info = data['sim_info']
post_proc_data = data['post-processing']
mast_pitch = data['data']['mast_pitch']
mast_diameter = data['data']['mast_diameter']
time = data['time']

print('time:', time)


#%% explore buckling data

# access data
buck_sim_name = list(sim_info.keys())[0]
buck_results = post_proc_data[buck_sim_name]
p_crit = buck_results['loads'][0]
coilable = buck_results['coilable'][0]

# print results
print('P_crit, sigma_crit, coilable:', p_crit, p_crit / mast_diameter, coilable)


#%% explore riks data

# access data
riks_sim_name = list(sim_info.keys())[1]
riks_results = post_proc_data[riks_sim_name]
u_3 = np.abs(np.array(riks_results['U'][-1]))
rf_3 = np.abs(np.array(riks_results['RF'][-1]))


# force-displacement
plt.figure()
plt.plot(u_3, rf_3)
plt.xlabel('Displacement /mm')
plt.ylabel('Force /N')

# stress-strain
plt.figure()
plt.plot(u_3 / mast_pitch, rf_3 / mast_diameter * 1e3)
plt.xlabel('Strain')
plt.ylabel('Stress /kPa')


plt.show()
