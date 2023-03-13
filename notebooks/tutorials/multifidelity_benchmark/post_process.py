import pandas as pd
import numpy as np
import f3dasm

fun_list = [fun.name for fun in f3dasm.functions.get_functions(d=1, randomized_term=False) if fun in f3dasm.functions.get_functions(d=500)]
print(fun_list)

for fun in fun_list:
    print(fun)
    for dim in range(1, 4):
        df_trial = []
        for trial in range(10):
            path = 'notebooks/tutorials/multifidelity_benchmark/outputs/Sogpr/2023-03-09/15-49-14/dim_%d/trial_%d/%s.csv' % (dim, trial, fun)
            df = pd.read_csv(path)
            df_trial.append(df['2'][3])
        print(np.mean(df_trial))
