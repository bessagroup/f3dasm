'''
Created on 2020-09-30 11:18:00
Last modified on 2020-09-30 11:43:43

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# third-party
from f3dasm.run.abaqus import run_sims

# local library
from abaqus_modules.get_results import get_results


# run

if __name__ == '__main__':

    example_name = 'example_0'

    run_sims(example_name, n_sims=4, abaqus_path='abaqus',
             keep_odb=True, pp_fnc=get_results, n_cpus=2)
