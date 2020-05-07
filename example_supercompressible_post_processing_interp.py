'''
Created on 2020-05-05 11:44:10
Last modified on 2020-05-06 00:16:53
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
'''


#%% imports

# standard library
import os

# third-party
from matplotlib import pyplot as plt

# local library
from f3das.misc.misc import get_int_number_from_str
from f3das.misc.file_handling import collect_folder_names
from f3das.post_processing.common import get_results
from f3das.post_processing.supercompressible import get_results_lin_buckle
from f3das.post_processing.supercompressible import read_and_clean_results_riks


#%% initialization

example_name = 'example_supercompressible_3d'
sim_dir = 'analyses'
additional_strain_thresh = .05
sim_numbers = None
# sim_numbers = range(0, 100)
# sim_numbers = [9]
plot = False


#%% computations

# colect folder names
dir_name = os.path.join(example_name, sim_dir)
folder_names = collect_folder_names(dir_name, sim_numbers=sim_numbers)

for folder_name in folder_names:

    # TODO: delete
    sim_number = get_int_number_from_str(folder_name)
    print('sim:', sim_number)

    # get data
    data = get_results(dir_name, folder_name)

    # get data (linear buckling)
    coilable, sigma_crit = get_results_lin_buckle(data)

    # get data (Riks)
    _, (strain, stress), (energy, (x, y)) = read_and_clean_results_riks(
        data, get_energy=True)

    print(coilable, sigma_crit, energy)

    # plot stress-strain
    if plot:
        plt.figure()
        plt.plot(strain, stress, label='Simul')
        if energy:
            plt.plot(x, y, label='Interp')
        plt.xlabel('Strain')
        plt.ylabel('Stress /kPa')
        plt.title('DoE point %i' % get_int_number_from_str(folder_name))
        plt.legend()

        plt.show()
