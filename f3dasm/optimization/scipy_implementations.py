from scipy.optimize import minimize, differential_evolution, dual_annealing
import numpy as np

from ..optimization.hyperparameters import (
    CG_Parameters,
    DifferentialEvolution_Parameters,
    DualAnnealing_Parameters,
    LBFGSB_Parameters,
    NelderMead_Parameters,
)

from ..base.optimization import Optimizer
from ..base.function import Function


class SciPyOptimizer(Optimizer):
    def _callback(self, xk: np.ndarray, *args, **kwargs) -> None:
        self.x_new.append(xk.tolist())

    def update_step(self) -> None:
        pass

    def run_algorithm(self, iterations: int, function: Function) -> None:
        pass

    def iterate(self, iterations: int, function: Function) -> None:
        self.x_new = []

        self.parameter.maxiter = iterations

        self.run_algorithm(iterations, function)

        self.x_new = np.array(self.x_new)

        # If x_new is empty, repeat best x0 to fill up total iteration
        if len(self.x_new) == 0:
            repeated_last_element = np.tile(
                self.data.get_n_best_input_parameters_numpy(nosamples=1).ravel(),
                (iterations - len(self.x_new), 1),
            )
            self.x_new = repeated_last_element

        # Repeat last iteration to fill up total iteration
        if len(self.x_new) < iterations:
            repeated_last_element = np.tile(self.x_new[-1], (iterations - len(self.x_new), 1))
            self.x_new = np.r_[self.x_new, repeated_last_element]

        self.data.add_numpy_arrays(input=self.x_new, output=function(self.x_new))


class SciPyMinimizeOptimizer(SciPyOptimizer):
    def run_algorithm(self, iterations: int, function: Function) -> None:
        minimize(
            fun=lambda x: function(x).item(),
            method=self.parameter.method,
            jac=lambda x: function.dfdx(x).ravel(),
            x0=self.data.get_n_best_input_parameters_numpy(nosamples=1).ravel(),
            callback=self._callback,
            options=self.parameter.__dict__,
            bounds=function.scale_bounds,
            tol=0.0,
        )


class CG(SciPyMinimizeOptimizer):
    """CG"""

    parameter: CG_Parameters = CG_Parameters()


class NelderMead(SciPyMinimizeOptimizer):
    """Nelder-Mead"""

    parameter: NelderMead_Parameters = NelderMead_Parameters()


class LBFGSB(SciPyMinimizeOptimizer):
    """L-BFGS-B"""

    parameter: LBFGSB_Parameters = LBFGSB_Parameters()


class DifferentialEvolution(SciPyOptimizer):
    """Differential Evolution"""

    parameter: DifferentialEvolution_Parameters = DifferentialEvolution_Parameters()

    def run_algorithm(self, iterations: int, function: Function) -> None:
        differential_evolution(
            func=lambda x: function(x).item(),
            bounds=function.scale_bounds,
            strategy=self.parameter.strategy,
            maxiter=iterations,
            popsize=self.parameter.population,
            tol=self.parameter.tol,
            mutation=self.parameter.mutation,
            recombination=self.parameter.recombination,
            seed=self.seed,
            callback=self._callback,
            polish=self.parameter.polish,
            init=self.data.get_n_best_input_parameters_numpy(nosamples=self.parameter.population),
            atol=self.parameter.atol,
            updating=self.parameter.updating,
        )


class DualAnnealing(SciPyOptimizer):
    """Dual Annealing"""

    parameter: DualAnnealing_Parameters = DualAnnealing_Parameters()

    def run_algorithm(self, iterations: int, function: Function) -> None:
        dual_annealing(
            func=lambda x: function(x).item(),
            bounds=function.scale_bounds,
            maxiter=iterations,
            initial_temp=self.parameter.initial_temp,
            restart_temp_ratio=self.parameter.restart_temp_ratio,
            visit=self.parameter.visit,
            accept=self.parameter.accept,
            maxfun=10000000.0,
            seed=self.seed,
            no_local_search=self.parameter.no_local_search,
            callback=self._callback,
            x0=self.data.get_n_best_input_parameters_numpy(nosamples=1).ravel(),
        )
