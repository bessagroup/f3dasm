'''
Created on 2020-05-05 10:55:47
Last modified on 2020-05-05 14:19:09
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Create gif with load-displacement or stress-strain pics. Assumes pictures were
already created.
'''


#%% imports

# standard library
import os

# third-party
import imageio


#%% initialization

# pics folder
pics_dir = 'example_supercompressible_3d_pics'
init_file_name = 'stress_strain'
duration_per_sim = 1.
sim_numbers = range(0, 10)


#%% create gif

# get all pics
file_names_temp = [name for name in os.listdir(pics_dir)
                   if name.startswith(init_file_name)]
existing_sim_numbers = [int(file_name.split('.')[0].split('_')[-1])
                        for file_name in file_names_temp]
indices = sorted(range(len(existing_sim_numbers)), key=existing_sim_numbers.__getitem__)

# update sim_numbers
if not sim_numbers:
    sim_numbers = existing_sim_numbers
    sim_numbers.sort()

# get images and create gif
file_names = [file_names_temp[index] for index in indices
              if existing_sim_numbers[index] in sim_numbers]
images = [imageio.imread(os.path.join(pics_dir, fn)) for fn in file_names]
gif_name = os.path.join('%s.gif' % init_file_name)
imageio.mimsave(gif_name, images, duration=duration_per_sim)
