from typing import List
from f3dasm.base.space import ConstantParameter

from f3dasm.base.utils import make_nd_continuous_design
from ..sampling import SobolSequence

from ..base.function import AugmentedFunction, Function, MultiFidelityFunction

import numpy as np

def create_analytical_mf_problem(
    base_fun: Function,
    dim: int,
    fids: List[float],
    costs: List[float],
    samp_nos: List[int],
    ):

    funs = []
    mf_design_space = []
    mf_sampler = []
    mf_train_data = []

    for fid_no, (fid, cost, samp_no) in enumerate(zip(fids, costs, samp_nos)):

        fun = AugmentedFunction(
                base_fun=base_fun,
                fid=fid,
                )
        
        parameter_DesignSpace = make_nd_continuous_design(
            bounds=np.tile([0.0, 1.0], (dim, 1)),
            dimensionality=dim,
        )
        fidelity_parameter = ConstantParameter(name="fid", constant_value=fid)
        parameter_DesignSpace.add_input_space(fidelity_parameter)

        sampler = SobolSequence(design=parameter_DesignSpace)

        init_train_data = sampler.get_samples(numsamples=samp_no)
        init_train_data.add_output(output=fun(init_train_data))

        funs.append(fun)
        mf_design_space.append(parameter_DesignSpace)
        mf_sampler.append(sampler)
        mf_train_data.append(init_train_data)

    mffun = MultiFidelityFunction(
        funs=funs,
        fids=fids,
        costs=costs,
    )

    return mf_train_data, mffun, mf_design_space, mf_sampler