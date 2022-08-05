from scipy.optimize import minimize, differential_evolution, dual_annealing
import numpy as np

from ..base.optimization import Optimizer
from ..base.function import Function


class SciPyGlobalOptimizer(Optimizer):
    def c(self, xk: np.ndarray, convergence) -> None:
        self.x_new.append(xk.tolist())

    def update_step(self) -> None:
        pass

    def run_algorithm(self, iterations: int, function: Function) -> None:
        pass

    def iterate(self, iterations: int, function: Function) -> None:
        self.x_new = []

        self.hyperparameters["maxiter"] = iterations

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

        self.data.add_numpy_arrays(input=self.x_new, output=function.eval(self.x_new))


class SciPyLocalOptimizer(Optimizer):
    def c(self, xk: np.ndarray) -> None:
        self.x_new.append(xk.tolist())

    def update_step(self) -> None:
        pass

    def iterate(self, iterations: int, function: Function) -> None:
        self.x_new = []

        self.hyperparameters["maxiter"] = iterations

        minimize(
            fun=lambda x: function.eval(x).item(),
            method=self.algorithm,
            jac=lambda x: function.dfdx(x).ravel(),
            x0=self.data.get_n_best_input_parameters_numpy(nosamples=1).ravel(),
            callback=self.c,
            options=self.hyperparameters,
            bounds=[(function.scale_bounds[0], function.scale_bounds[1]) for _ in range(function.dimensionality)],
            tol=0.0,
        )

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

        self.data.add_numpy_arrays(input=self.x_new, output=function.eval(self.x_new))


class CG(SciPyLocalOptimizer):
    """CG"""

    def init_parameters(self):
        # Default hyperparameters
        self.defaults = {"gtol": 0.0}

    def set_algorithm(self):
        self.algorithm = "CG"


class NelderMead(SciPyLocalOptimizer):
    """Nelder-Mead"""

    def init_parameters(self):
        # Default hyperparameters
        self.defaults = {"xatol": 0.0, "fatol": 0.0, "adaptive": False}

    def set_algorithm(self):
        self.algorithm = "Nelder-Mead"


class LBFGSB(SciPyLocalOptimizer):
    """L-BFGS-B"""

    def init_parameters(self):
        # Default hyperparameters
        self.defaults = {"ftol": 0.0, "gtol": 0.0}

    def set_algorithm(self):
        self.algorithm = "L-BFGS-B"


class DifferentialEvolution(SciPyGlobalOptimizer):
    """Differential Evolution"""

    def init_parameters(self):
        # Default hyperparameters
        self.defaults = {
            "strategy": "best1bin",
            "population": 15,
            "tol": 0.0,
            "mutation": (0.5, 1),
            "recombination": 0.7,
            "polish": False,
            "atol": 0.0,
            "updating": "immediate",
        }

    def run_algorithm(self, iterations: int, function: Function) -> None:
        differential_evolution(
            func=lambda x: function.eval(x).item(),
            bounds=[(function.scale_bounds[0], function.scale_bounds[1]) for _ in range(function.dimensionality)],
            strategy=self.hyperparameters["strategy"],
            maxiter=iterations,
            popsize=self.hyperparameters["population"],
            tol=self.hyperparameters["tol"],
            mutation=self.hyperparameters["mutation"],
            recombination=self.hyperparameters["recombination"],
            seed=self.seed,
            callback=self.c,
            polish=self.hyperparameters["polish"],
            init=self.data.get_n_best_input_parameters_numpy(nosamples=self.hyperparameters["population"]),
            atol=self.hyperparameters["atol"],
            updating=self.hyperparameters["updating"],
        )

    def set_algorithm(self):
        pass


class DualAnnealing(SciPyGlobalOptimizer):
    """Differential Evolution"""

    def init_parameters(self):
        # Default hyperparameters
        self.defaults = {
            "initial_temp": 5230.0,
            "restart_temp_ratio": 2e-05,
            "visit": 2.62,
            "accept": -5.0,
            "no_local_search": False,
            "population": 1,
        }

    def c(self, x: np.ndarray, f: float, context: float) -> None:
        self.x_new.append(x.tolist())

    def run_algorithm(self, iterations: int, function: Function) -> None:
        dual_annealing(
            func=lambda x: function.eval(x).item(),
            bounds=[(function.scale_bounds[0], function.scale_bounds[1]) for _ in range(function.dimensionality)],
            maxiter=iterations,
            initial_temp=self.hyperparameters["initial_temp"],
            restart_temp_ratio=self.hyperparameters["restart_temp_ratio"],
            visit=self.hyperparameters["visit"],
            accept=self.hyperparameters["accept"],
            maxfun=10000000.0,
            seed=self.seed,
            no_local_search=self.hyperparameters["no_local_search"],
            callback=self.c,
            x0=self.data.get_n_best_input_parameters_numpy(nosamples=1).ravel(),
        )

    def set_algorithm(self):
        pass
