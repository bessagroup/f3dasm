import f3dasm
import numpy as np

base_fun = f3dasm.functions.AlpineN2(
    dimensionality=1,
    scale_bounds=np.tile([0.0, 1.0], (1, 1)),
    offset=False,
    )

from f3dasm.functions.fidelity_augmentors import Scale

# import itertools
# import pandas as pd

# a = [1, 2, 3]
# b = ['x', 'y', 'z']
# c = ['foo', 'bar', 'baz']

# # df = pd.DataFrame(columns=['c1', 'c2', 'c3'])

# # rows_list = []
# # for row in itertools.product(*[a, b, c]):
# #     rows_list.append(row)

# df = pd.DataFrame(
#     columns=['c1', 'c2', 'c3'], 
#     data=list(itertools.product(*[a, b, c])))

# print(df)