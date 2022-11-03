import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import f3dasm
from f3dasm import AugmentedFunction, Data
from f3dasm.base import function

dim = 1

base_fun = f3dasm.functions.Ackley(dimensionality=dim)
fids = [0.5, 1]
costs = [0.5, 1]
samp_nos = [20, 10]
funs = []
for fid_no, fid in enumerate(fids):
    funs.append(
        AugmentedFunction(
            base_fun=base_fun,
            fid=fid,
        )
    )

mffun = function.MultiFidelityFunction(
    funs=funs,
    fids=fids,
    costs=costs,
)

parameter_DesignSpace = f3dasm.make_nd_continuous_design(
    bounds=base_fun.input_domain.astype(float),
    dimensionality=dim,
)

SobolSampler = f3dasm.sampling.SobolSequenceSampling(design=parameter_DesignSpace)

optimizer = f3dasm.optimization.MFBayesianOptimizationTorch(
    # data=mfsamples,
    data=Data(design=parameter_DesignSpace),
    mffun=mffun,
)

# mfsamples = []
# for fid_no, samp_no in enumerate(samp_nos):
#     samples = SobolSampler.get_samples(numsamples=samp_no)
#
#     samples.add_output(output=mffun.funs[fid_no](samples))
#
#     mfsamples.append(samples)
#
# optimizer.set_data(mfsamples)

optimizer.init_parameters()

res = f3dasm.run_mf_optimization(
    optimizer=optimizer,
    mffunction=mffun,
    sampler=SobolSampler, # samples just in the design space
    iterations=10,
    seed=123,
    number_of_samples=samp_nos,
    # budget=10 # add budget to optimization iterator
)

print(res)