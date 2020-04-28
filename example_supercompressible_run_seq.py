'''
Created on 2020-04-26 04:41:27
Last modified on 2020-04-28 17:17:36
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Run supercompressible.
'''


#%% imports

# standard library
import multiprocessing as mp

# local library
from f3das.abaqus.run.misc import run_simuls_sequentially
from f3das.abaqus.run.misc import get_missing_simuls


#%% initialization

n_cpus = 3

# folder structure
example_name = 'example_supercompressible_7d'
simuls_dir_name = 'analyses'

# points to run
# points = list(range(5000))
points = get_missing_simuls(example_name, simuls_dir_name)


#%% run abaqus

# distribute points
points = sorted(points)
points_cpus = []
for i in range(n_cpus):
    points_cpus.append(points[i::n_cpus])


if __name__ == '__main__':

    # start pool
    pool = mp.Pool(n_cpus)

    # run simuls
    for i, points in enumerate(points_cpus):
        wait_time = i * 5
        pool.apply_async(run_simuls_sequentially,
                         args=(example_name, simuls_dir_name, points,
                               wait_time))

    # close pool and wait process completion
    pool.close()
    pool.join()
