#                                                                       Modules
# =============================================================================

# Third-party
import numpy as np
import torch
from botorch.test_functions import SyntheticTestFunction
from torch import Tensor

# Locals
from ...functions.pybenchfunction import PyBenchFunction

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Leo Guo (L.Guo@tudelft.nl)'
__credits__ = ['Leo Guo']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


# from MFBO import Comsol_Sim_low, Comsol_Sim_high


tkwargs = {
    "dtype": torch.double,
    # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
}


class botorch_TestFunction(SyntheticTestFunction):
    def __init__(self, fun: PyBenchFunction, negate=False):
        self.name = fun.name
        self.continuous = fun.continuous
        self.convex = fun.convex
        self.separable = fun.separable
        self.differentiable = fun.differentiable
        self.multimodal = fun.multimodal
        self.randomized_term = fun.randomized_term
        self.parametric = fun.parametric

        self.fun = fun
        self.dim = fun.dimensionality
        self._bounds = fun.input_domain
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        # self.negate = negate
        # super().__init__(negate=negate)
        super().__init__()

    def evaluate_true(self, X: Tensor) -> Tensor:
        res = torch.Tensor(np.apply_along_axis(self.fun, 1, X.cpu()))

        # if self.negate:
        #     res = -res

        return res


class AugmentedTestFunction(SyntheticTestFunction):
    def __init__(self, fun, negate=False, noise_type="bn"):
        self.name = "Augmented" + fun.name.replace(" ", "")
        self.continuous = fun.continuous
        self.convex = fun.convex
        self.separable = fun.separable
        self.differentiable = fun.differentiable
        self.multimodal = fun.multimodal
        self.randomized_term = fun.randomized_term
        self.parametric = fun.parametric

        self.fun = fun.evaluate_true
        self.opt = fun.fun.get_global_minimum
        self.dim = fun.dim + 1
        self._bounds = fun._bounds
        self._optimizers = fun._optimizers

        self.noise_type = noise_type
        self.negate = negate

        super().__init__(negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        # torch.random.manual_seed(123)

        res_high = self.fun(X[:, :-1]).flatten().to(**tkwargs)

        fid = X[:, -1].to(**tkwargs)
        # print(fid)
        # print(fid.shape)
        white_noise = torch.normal(0, 1, size=(len(res_high),))

        stdev = 1
        if len(res_high) > 1:
            stdev = torch.std(res_high)

        if self.noise_type == "bn":
            # res_low = stdev * white_noise + torch.mean(res_high) #+ 500 * brown_noise
            res_low = stdev * white_noise  # + 500 * brown_noise
        elif self.noise_type == "n":
            res_low = stdev * white_noise + res_high
        elif self.noise_type == "b":
            # res_low = torch.mean(res_high)
            res_low = 0
        else:
            res_low = stdev * white_noise + \
                torch.mean(res_high)  # + 500 * brown_noise

        # Noise ideas ###
        # noise = 2 * (torch.rand(res_high.shape) - 0.5)
        # white_noise = white_noise_pre[:len(res_high)]
        # brown_noise = torch.cumsum(white_noise, dim=-1)
        # res_low = torch.mean(res_high) #+ 500 * brown_noise
        # res_low = self.fun(X[:, :-1] - 5).flatten()
        # res_low = res_high * 1.25 #+ torch.mean(res_high) #+ 500 * brown_noise
        # res_low = np.sqrt(bds[0][1] - bds[0][0]) * brown_noise + torch.mean(res_high)

        c = 1

        res = fid**c * res_high + (1 - fid**c) * res_low
        # print(res)

        # if self.negate:
        #     res = -res

        return res


# class ComsolTestFunction(SyntheticTestFunction):
#     def __init__(self):
#         self.name = 'comsol'
#         # self._bounds = [(0, 6)]
#         # self._optimizers = [(1.5 * np.pi, -1)]
#         # self.dim = 2
#         # self._bounds = [(-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2)]
#         # self._optimizers = [(0, 0, 0, 0, 0, 0, 0, 0, 0)]
#         # self._bounds = [(-2, 2), (-2, 2),]
#         # self._optimizers = [(0, 0,)]
#         # self.dim = 3
#         self._bounds = [(0, 1), (0, 1), ]
#         self._optimizers = [(0, 0)]
#         self.dim = 3
#         self.noise_type = 'NA'
#         self.negate = False
#         super().__init__()
#
#     def evaluate_true(self, X: Tensor) -> Tensor:
#         # torch.random.manual_seed(123)
#
#         fid = X[:, -1].to(**tkwargs)
#
#         # res = fid ** c * res_high + (1 - fid ** c) * res_low
#         res = torch.zeros_like(fid)
#
#         for i, x_i in enumerate(X):
#             if fid[i] == 1:
#                 res[i] = torch.tensor(Comsol_Sim_high(x_i[:-1].numpy()))
#             else:
#                 res[i] = torch.tensor(Comsol_Sim_low(x_i[:-1].numpy()))
#
#         if self.negate:
#             res = -res
#
#         return res
