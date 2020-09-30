'''
Created on 2020-09-24 14:35:34
Last modified on 2020-09-24 15:09:39

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# third-party
import numpy as np
from SALib.sample import sobol_sequence, saltelli, latin
import pandas as pd


# object definition

def create_doe(n, doe_variables, sample_strat='sobol', seed=None):
    '''
    Parameters
    ----------
    sample_strat : str
        Possible values are `sobol`, `saltelli`, `latin`, `random`.
    '''

    # initialization
    sample_poss = {'sobol': _sobol_sampling,
                   'saltelli': _saltelli_sampling,
                   'latin': _latin_sampling,
                   'random': _random_sampling}
    sample = sample_poss[sample_strat]

    # get points
    points = sample(n, doe_variables, seed=seed)
    doe_variables_ls = list(doe_variables.keys())
    points_pd = pd.DataFrame(points, columns=doe_variables_ls)

    return points_pd


def _sobol_sampling(n, doe_variables, **kwargs):
    dim = len(doe_variables)
    points = sobol_sequence.sample(n, dim)
    for i, bounds in enumerate(doe_variables.values()):
        points[:, i] = points[:, i] * (bounds[1] - bounds[0]) + bounds[0]

    return points


def _saltelli_sampling(n, doe_variables, seed):
    dim = len(doe_variables)
    N = int(n / (2 * dim + 2))
    problem = _define_salib_problem(doe_variables)
    points = saltelli.sample(problem, N, seed=seed)

    return points


def _latin_sampling(n, doe_variables, seed):
    problem = _define_salib_problem(doe_variables)
    points = latin.sample(problem, n, seed=seed)

    return points


def _random_sampling(n, doe_variables, seed):
    np.random.seed(seed)
    dim = len(doe_variables)
    points = np.random.random((n, dim))
    for i, bounds in enumerate(doe_variables.values()):
        points[:, i] = points[:, i] * (bounds[1] - bounds[0]) + bounds[0]

    return points


def _define_salib_problem(doe_variables):
    problem = {'num_vars': len(doe_variables),
               'names': doe_variables.keys(),
               'bounds': [value for value in doe_variables.values()]}

    return problem
