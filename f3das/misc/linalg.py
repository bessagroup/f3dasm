'''
Created on 2020-03-30 11:12:12
Last modified on 2020-04-07 17:04:26
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Define matrix-related functions.

Notes
-----
-Care must be taken with imports to make sure this module can be used inside
Abaqus.
-scipy is not used because it does not exit in Abaqus.
'''


#%% imports

# third-party
import numpy as np


#%% function definition

def symmetricize_vector(vec):
    '''
    Create a symmetric matrix given a vector.

    Parameters
    ----------
    vec : array-like
        Elements ordered by its position in the upper triangular matrix.
    '''

    # compute dimension
    n = int(1 / 2 * (-1 + np.sqrt(1 + 8 * len(vec))))

    # assign variables to the right positions
    matrix = np.empty((n, n))
    k = 0
    for i in range(n):
        for j in range(i, n):
            matrix[i, j] = vec[k]
            if i != j:
                matrix[j, i] = matrix[i, j]
            k += 1

    return matrix


def sqrtm(matrix):
    '''
    Square root of a square symmetric matrix.

    Parameters
    ----------
    matrix : array-like. shape: [n, n]
        Symmetric matrix
    '''

    # get eigenvalues and vectors
    w, v = np.linalg.eig(matrix)

    # define spectral decomposition
    spec_decomp = np.zeros(np.shape(matrix))
    for i in range(np.size(w)):
        spec_decomp[i, i] = w[i]
    spec_decomp_sqrt = np.sqrt(spec_decomp)

    # square root of the original matrix
    matrix_sqrt = np.matmul(v, np.matmul(spec_decomp_sqrt, v.transpose()))

    return matrix_sqrt
