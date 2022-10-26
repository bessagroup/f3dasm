from dataclasses import dataclass
import pygmo as pg


from ..base.optimization import OptimizerParameters
from .adapters.pygmo_implementations import PygmoAlgorithm

# # from dataclasses import dataclass
# # from typing import Set
# # from scipy.optimize import differential_evolution

# # from .adapters.scipy_implementations import SciPyOptimizer
# # from ..base.function import Function
# # from ..base.optimization import OptimizerParameters


# # @dataclass
# # class DifferentialEvolution_Parameters(OptimizerParameters):
# #     """Hyperparameters for DifferentialEvolution optimizer"""

# #     strategy: str = "best1bin"
# #     population: int = 15
# #     tol: float = 0.0
# #     mutation: Set = (0.5, 1)
# #     recombination: float = 0.7
# #     polish: bool = False
# #     atol: float = 0.0
# #     updating: str = "immediate"


# # class DifferentialEvolution(SciPyOptimizer):
# #     """Differential Evolution"""

# #     parameter: DifferentialEvolution_Parameters = DifferentialEvolution_Parameters()

# #     def run_algorithm(self, iterations: int, function: Function) -> None:
# #         differential_evolution(
# #             func=lambda x: function(x).item(),
# #             bounds=function.scale_bounds,
# #             strategy=self.parameter.strategy,
# #             maxiter=iterations,
# #             popsize=self.parameter.population,
# #             tol=self.parameter.tol,
# #             mutation=self.parameter.mutation,
# #             recombination=self.parameter.recombination,
# #             seed=self.seed,
# #             callback=self._callback,
# #             polish=self.parameter.polish,
# #             init=self.data.get_n_best_input_parameters_numpy(nosamples=self.parameter.population),
# #             atol=self.parameter.atol,
# #             updating=self.parameter.updating,
# #         )


@dataclass
class DifferentialEvolution_Parameters(OptimizerParameters):
    """Hyperparameters for DifferentialEvolution optimizer"""

    population: int = 30
    gen: int = 1
    F: float = 0.8
    CR: float = 0.9
    variant: int = 2
    ftol: float = 0.0
    xtol: float = 0.0


class DifferentialEvolution(PygmoAlgorithm):
    "DifferentialEvolution optimizer implemented from pygmo"

    parameter: DifferentialEvolution_Parameters = DifferentialEvolution_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.de(
                gen=self.parameter.gen,
                F=self.parameter.F,
                CR=self.parameter.CR,
                variant=self.parameter.variant,
                ftol=self.parameter.ftol,
                xtol=self.parameter.xtol,
                seed=self.seed,
            )
        )
