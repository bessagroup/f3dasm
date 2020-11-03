'''
Created on 2020-11-03 10:32:01
Last modified on 2020-11-03 10:44:29

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# third-party
import numpy as np
from f3dasm.utils.linalg import symmetricize_vector
from f3dasm.utils.solid_mechanics import compute_small_strains_from_green


# computations
E_vec = np.array([1., 0.2, -0.1])
E = symmetricize_vector(E_vec)

eps = compute_small_strains_from_green(E)
print(eps)
