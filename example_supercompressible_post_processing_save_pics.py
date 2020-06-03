'''
Created on 2020-05-04 18:35:56
Last modified on 2020-05-11 19:59:45
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Save pictures of the load-displacement and stress-strain curves.
'''


#%% imports

# standard library
import os

# third-party
from matplotlib import pyplot as plt

# local library
from f3das.misc.misc import get_int_number_from_str
from f3das.misc.file_handling import collect_folder_names
from f3das.post_processing.supercompressible import read_and_clean_results_riks
from f3das.post_processing.common import get_results

# TODO: update


#%% initialization

# pics to save
# sim_numbers = list(range(10))  # None means all
sim_numbers = [0, 11, 12, 14, 16, 28, 36, 40, 60, 68, 72, 88, 96, 108, 112]

# pics folder
pics_dir = 'example_supercompressible_3d_pics_error'

# results folder structure
example_name = 'example_supercompressible_3d'
sim_dir = 'analyses'


#%% create dir and pics

# create dir
os.mkdir(pics_dir)

# colect folder names
dir_name = os.path.join(example_name, sim_dir)
folder_names = collect_folder_names(dir_name, sim_numbers=sim_numbers)

# save pics
for folder_name in folder_names:

    sim_num = get_int_number_from_str(folder_name)

    # get data
    data = get_results(dir_name, folder_name)
    (u_3, rf_3), (strain, stress), _, _ = read_and_clean_results_riks(
        data, get_energy=False)

    # force-displacement
    plt.figure()
    plt.plot(u_3, rf_3)
    plt.xlabel('Displacement /mm')
    plt.ylabel('Force /N')
    plt.title('DoE point %i' % sim_num)
    plt.savefig(os.path.join(pics_dir, 'force_displacement_%i' % sim_num))
    plt.close()

    # stress-strain
    plt.figure()
    plt.plot(strain, stress)
    plt.xlabel('Strain')
    plt.ylabel('Stress /kPa')
    plt.title('DoE point %i' % sim_num)
    plt.savefig(os.path.join(pics_dir, 'stress_strain_%i' % sim_num))
    plt.close()
