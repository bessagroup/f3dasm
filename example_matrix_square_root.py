'''
Created on 2020-04-07 16:21:24
Last modified on 2020-04-07 17:10:41
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Show that the square root of a matrix is working properly.

Notes
-----
-scipy coulb be used in the scripts, but it does not exist in Abaqus.
'''


#%% imports

# third-party
import numpy as np

# local library
from src.misc.linalg import symmetricize_vector
from src.misc.linalg import sqrtm


#%% initialization

a = [1.831, 0.731, 0.985]
b = [2.031, 0.162, 2.021, 1.245, 0, 2.561]


#%% computations

# get matrices
A = symmetricize_vector(a)
B = symmetricize_vector(b)

# get matrices square roots
A_sqrt = sqrtm(A)
B_sqrt = sqrtm(B)


#%% print results

print('2D:')
print('A:', A)
print('sqrtm(A):', A_sqrt)
print('verification:', A - np.matmul(A_sqrt, A_sqrt))
print('3D:')
print('B:', B)
print('sqrtm(B):', B_sqrt)
print('verification:', B - np.matmul(B_sqrt, B_sqrt))
