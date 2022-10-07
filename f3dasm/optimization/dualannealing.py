# from dataclasses import dataclass
# from scipy.optimize import dual_annealing

# from ..base.function import Function
# from ..base.optimization import OptimizerParameters
# from .adapters.scipy_implementations import SciPyOptimizer


# @dataclass
# class DualAnnealing_Parameters(OptimizerParameters):
#     """Hyperparameters for DualAnnealing optimizer"""

#     initial_temp: float = 5230.0
#     restart_temp_ratio: float = 2e-05
#     visit: float = 2.62
#     accept: float = -5.0
#     no_local_search: bool = False


# class DualAnnealing(SciPyOptimizer):
#     """Dual Annealing"""

#     parameter: DualAnnealing_Parameters = DualAnnealing_Parameters()

#     def run_algorithm(self, iterations: int, function: Function) -> None:
#         dual_annealing(
#             func=lambda x: function(x).item(),
#             bounds=function.scale_bounds,
#             maxiter=iterations,
#             initial_temp=self.parameter.initial_temp,
#             restart_temp_ratio=self.parameter.restart_temp_ratio,
#             visit=self.parameter.visit,
#             accept=self.parameter.accept,
#             maxfun=10000000.0,
#             seed=self.seed,
#             no_local_search=self.parameter.no_local_search,
#             callback=self._callback,
#             x0=self.data.get_n_best_input_parameters_numpy(nosamples=1).ravel(),
#         )
