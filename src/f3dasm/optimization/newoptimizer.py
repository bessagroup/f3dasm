# Standard
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

import autograd.core
import autograd.numpy as np
from autograd import elementwise_grad as egrad

# Locals
from f3dasm._imports import try_import
from f3dasm.datageneration.functions import Function
from f3dasm.optimization.optimizer import Optimizer

from ..datageneration.datagenerator import DataGenerator
from ..design.domain import Domain
from ..design.experimentdata import (ExperimentData, ExperimentSample,
                                     _number_of_overiterations,
                                     _number_of_updates)
from .optimizer import OptimizerParameters

# Third-party extension
with try_import('optimization') as _imports:
    import jax
    import nevergrad as ng
    import pygmo as pg
    import tensorflow as tf
    from evosax import CMA_ES, Strategy
    from jax._src.typing import Array
    from keras import Model

from scipy.optimize import minimize


class NewOptimizer:
    type: ClassVar[str] = 'any'
    hyperparameters: OptimizerParameters = OptimizerParameters()

    def __init__(self, domain: Domain, hyperparameters: Optional[Dict[str, Any]] = None,
                 seed: Optional[int] = None, name: Optional[str] = None):
        """Optimizer class for the optimization of a data-driven process

        Parameters
        ----------
        domain : Domain
            Domain indicating the search-space of the optimization parameters
        hyperparameters : Optional[Dict[str, Any]], optional
            Hyperparameters of the optimizer, by default None, it will use the default hyperparameters
        seed : Optional[int], optional
            Seed of the random number generator for stochastic optimization processes, by default None, set to random
        name : Optional[str], optional
            Name of the optimization object, by default None, it will use the name of the class
        """
        # Create an empty dictionary when hyperparameters is None
        if hyperparameters is None:
            hyperparameters = {}

        # Overwrite the default hyperparameters with the given hyperparameters
        self.hyperparameters.__init__(**hyperparameters)

        # Set the name of the optimizer to the class name if no name is given
        if name is None:
            name = self.__class__.__name__

        # Set the seed to a random number if no seed is given
        if seed is None:
            seed = np.random.randint(low=0, high=1e5)

        self.domain = domain
        self.seed = seed
        self.name = name
        self.__post_init__()

    def __post_init__(self):
        self._check_imports()
        self.set_seed()
        self.init_data()
        self.set_algorithm()

    @staticmethod
    def _check_imports():
        ...

    def init_data(self):
        """Set the data atrribute to an empty ExperimentData object"""
        self.data = ExperimentData(self.domain)

    def set_algorithm(self):
        """Set the algorithm attribute to the algorithm of choice"""
        ...

    def _construct_model(self, data_generator: DataGenerator):
        ...

    def _check_number_of_datapoints(self):
        """Check if the number of datapoints is sufficient for the initial population

        Raises
        ------
        ValueError
            Raises then the number of datapoints is insufficient
        """
        if len(self.data) < self.hyperparameters.population:
            raise ValueError(
                f'There are {len(self.data)} datapoints available, \
                     need {self.hyperparameters.population} for initial population!'
            )

    def set_seed(self):
        """Set the seed of the random number generator"""
        ...

    def reset(self):
        """Reset the optimizer to its initial state"""
        self.__post_init__()

    def set_data(self, data: ExperimentData):
        """Set the data attribute to the given data"""
        self.data = data

    def set_x0(self, experiment_data: ExperimentData):
        """Set the initial population to the best n samples of the given data

        Parameters
        ----------
        experiment_data : ExperimentData
            Data to be used for the initial population

        """
        x0 = experiment_data.get_n_best_output(self.hyperparameters.population)
        x0.reset_index()
        self.data = x0

    def get_name(self) -> str:
        """Get the name of the optimizer

        Returns
        -------
        str
            name of the optimizer
        """
        return self.name

    def get_info(self) -> List[str]:
        """Give a list of characteristic features of this optimizer

        Returns
        -------
            List of strings denoting the characteristics of this optimizer
        """
        return []

    def update_step(self, data_generator: DataGenerator) -> ExperimentData:
        """Update step of the optimizer. Needs to be implemented by the child class

        Parameters
        ----------
        data_generator : DataGenerator
            data generator object to calculate the objective value

        Returns
        -------
        ExperimentData
            ExperimentData object containing the new samples

        Raises
        ------
        NotImplementedError
            Raises when the method is not implemented by the child class
        """
        raise NotImplementedError(
            "You should implement an update step for your algorithm!")


class PartialExperimentData(ExperimentData):
    def _iterate(self, optimizer: NewOptimizer, data_generator: DataGenerator,
                 iterations: int, kwargs: Optional[dict] = None):

        optimizer.set_x0(self)
        optimizer._check_number_of_datapoints()

        optimizer._construct_model(data_generator)

        for _ in range(_number_of_updates(iterations, population=optimizer.hyperparameters.population)):
            new_samples = optimizer.update_step(function=data_generator)
            self.add_experiments(new_samples)

            # If applicable, evaluate the new designs:
            self.run(data_generator.run, mode='sequential', kwargs=kwargs)

            optimizer.set_data(self)

        # Remove overiterations
        self.remove_rows_bottom(_number_of_overiterations(
            iterations, population=optimizer.hyperparameters.population))

        # Reset the optimizer
        optimizer.reset()


class _PygmoProblem:
    """Convert a testproblem from the problemset to pygmo object

    Parameters
    ----------
    domain
        domain to be used
    func
        function to be evaluated
    seed
        seed for the random number generator
        _description_
    """

    def __init__(self, domain: Domain, func: DataGenerator, seed: Optional[int] = None):
        self.domain = domain
        self.func = func
        self.seed = seed

        if self.seed is not None:
            pg.set_global_rng_seed(self.seed)

    def fitness(self, x: np.ndarray) -> np.ndarray:
        """Pygmo representation of returning the objective value of a function

        Parameters
        ----------
        x
            input vector

        Returns
        -------
            fitness
        """
        evaluated_sample: ExperimentSample = self.func.run(ExperimentSample.from_numpy(x))
        _, y_ = evaluated_sample.to_numpy()
        return y_.ravel()  # pygmo doc: should output 1D numpy array

    def batch_fitness(self, x: np.ndarray) -> np.ndarray:
        """Pygmo representation of returning multiple objective values of a function

        Parameters
        ----------
        x
            input vectors

        Returns
        -------
            fitnesses
        """
        # Pygmo representation of returning multiple objective values of a function
        return self.fitness(x)

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        """Box-constrained boundaries of the problem. Necessary for pygmo library

        Returns
        -------
            box constraints
        """
        return (
            [parameter.lower_bound for parameter in self.domain.get_continuous_parameters().values()],
            [parameter.upper_bound for parameter in self.domain.get_continuous_parameters().values()],
        )


class PygmoAlgorithm(NewOptimizer):
    """Wrapper around the pygmo algorithm class

    Parameters
    ----------
    data
        ExperimentData-object
    hyperparameters
        Dictionary with hyperparameters
    seed
        seed to set the optimizer
    defaults
        Default hyperparameter arguments
    """

    @staticmethod
    def _check_imports():
        _imports.check()

    def set_seed(self):
        """Set the seed for pygmo

        Parameters
        ----------
        seed
            seed for the random number generator
        """
        pg.set_global_rng_seed(seed=self.seed)

    def update_step(self, data_generator: DataGenerator) -> ExperimentData:
        """Update step of the algorithm

        Parameters
        ----------
        function
            function to be evaluated

        Returns
        -------
            tuple of updated input parameters (x) and objecti value (y)
        """
        # Construct the PygmoProblem
        prob = pg.problem(
            _PygmoProblem(
                domain=self.domain,
                func=data_generator,
                seed=self.seed,
            )
        )

        # Construct the population
        pop = pg.population(prob, size=self.hyperparameters.population)

        # Set the population to the latest datapoints
        pop_x = self.data.get_input_data(
        ).iloc[-self.hyperparameters.population:].to_numpy()
        pop_fx = self.data.get_output_data(
        ).iloc[-self.hyperparameters.population:].to_numpy()

        for index, (x, fx) in enumerate(zip(pop_x, pop_fx)):
            pop.set_xf(index, x, fx)

        # Iterate one step
        pop = self.algorithm.evolve(pop)

        # return the data
        return ExperimentData.from_numpy(domain=self.domain,
                                         input_array=pop.get_x(),
                                         output_array=pop.get_f())


@dataclass
class CMAES_Parameters(OptimizerParameters):
    """Hyperparameters for CMAES optimizer"""

    population: int = 30


class CMAES(PygmoAlgorithm):
    """Covariance Matrix Adaptation Evolution Strategy optimizer implemented from pygmo"""

    hyperparameters: CMAES_Parameters = CMAES_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.cmaes(
                gen=1,
                memory=True,
                seed=self.seed,
                force_bounds=self.hyperparameters.force_bounds,
            )
        )

    def get_info(self) -> List[str]:
        return ['Stable', 'Global', 'Population-Based']


class TensorflowOptimizer(NewOptimizer):
    @staticmethod
    def _check_imports():
        _imports.check()

    def update_step(self, data_generator: DataGenerator) -> ExperimentData:
        with tf.GradientTape() as tape:
            tape.watch(self.args["tvars"])
            logits = 0.0 + tf.cast(self.args["model"](None), tf.float64)  # tf.float32
            loss = self.args["func"](tf.reshape(
                logits, (len(self.domain))))

        grads = tape.gradient(loss, self.args["tvars"])
        self.algorithm.apply_gradients(zip(grads, self.args["tvars"]))

        x = logits.numpy().copy()
        y = loss.numpy().copy()

        # return the data
        return ExperimentData.from_numpy(domain=self.domain,
                                         input_array=x,
                                         output_array=np.atleast_2d(np.array(y)))

    def _construct_model(self, data_generator: DataGenerator):
        self.args = {}

        def fitness(x: np.ndarray) -> np.ndarray:
            evaluated_sample: ExperimentSample = data_generator.run(ExperimentSample.from_numpy(x))
            _, y_ = evaluated_sample.to_numpy()
            return y_

        self.args["model"] = _SimpelModel(
            None,
            args={
                "dim": len(self.domain),
                "x0": self.data.get_n_best_input_parameters_numpy(self.hyperparameters.population),
                "bounds": self.domain.get_bounds(),
            },
        )  # Build the model
        self.args["tvars"] = self.args["model"].trainable_variables

        # TODO: This is an important conversion!!
        self.args["func"] = _convert_autograd_to_tensorflow(data_generator.__call__)


def _convert_autograd_to_tensorflow(func: Callable):
    """Convert autograd function to tensorflow function

    Parameters
    ----------
    func
        callable function to convert

    Returns
    -------
        wrapper to convert autograd function to tensorflow function
    """
    @tf.custom_gradient
    def wrapper(x, *args, **kwargs):
        vjp, ans = autograd.core.make_vjp(func, x.numpy())

        def first_grad(dy):
            @tf.custom_gradient
            def jacobian(a):
                vjp2, ans2 = autograd.core.make_vjp(egrad(func), a.numpy())
                return ans2, vjp2  # hessian

            return dy * jacobian(x)

        return ans, first_grad

    return wrapper


class _Model(Model):
    def __init__(self, seed=None, args=None):
        super().__init__()
        self.seed = seed
        self.env = args


class _SimpelModel(_Model):
    """
    The class for performing optimization in the input space of the functions.
    """

    def __init__(self, seed=None, args=None):
        super().__init__(seed)
        self.z = tf.Variable(
            args["x0"],
            trainable=True,
            dtype=tf.float32,
            constraint=lambda x: tf.clip_by_value(
                x,
                clip_value_min=args["bounds"][:, 0],
                clip_value_max=args["bounds"][:, 1],
            ),
        )  # S:ADDED

    def call(self, inputs=None):
        return self.z


@dataclass
class Adam_Parameters(OptimizerParameters):
    """Hyperparameters for Adam optimizer"""

    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07
    amsgrad: bool = False


class Adam(TensorflowOptimizer):
    """Adam"""

    hyperparameters: Adam_Parameters = Adam_Parameters()

    def set_algorithm(self):
        self.algorithm = tf.keras.optimizers.Adam(
            learning_rate=self.hyperparameters.learning_rate,
            beta_1=self.hyperparameters.beta_1,
            beta_2=self.hyperparameters.beta_2,
            epsilon=self.hyperparameters.epsilon,
            amsgrad=self.hyperparameters.amsgrad,
        )

    def get_info(self) -> List[str]:
        return ['Stable', 'Global', 'First-Order', 'Single-Solution']


class EvoSaxOptimizer(NewOptimizer):
    type: str = 'evosax'
    # evosax_algorithm: Strategy = None

    def _construct_model(self, data_generator: DataGenerator):
        self.algorithm: Strategy = self.evosax_algorithm(
            num_dims=len(self.domain), popsize=self.hyperparameters.population)
        self.evosax_param = self.algorithm.default_params
        self.evosax_param = self.evosax_param.replace(clip_min=self.data.domain.get_bounds()[
            0, 0], clip_max=self.data.domain.get_bounds()[0, 1])

        self.state = self.algorithm.initialize(self.seed, self.evosax_param)

        x_init, y_init = self.data.get_n_best_output(self.hyperparameters.population).to_numpy()

        self.state = self.algorithm.tell(x_init, y_init.ravel(), self.state, self.evosax_param)

    def set_seed(self) -> None:
        self.seed = jax.random.PRNGKey(self.seed)

    def reset(self):
        self._check_imports()
        self.set_algorithm()

    def update_step(self, data_generator: DataGenerator) -> ExperimentData:
        _, rng_ask = jax.random.split(self.seed)

        # Ask for a set candidates
        x, state = self.algorithm.ask(rng_ask, self.state, self.evosax_param)

        # Evaluate the candidates
        x_experimentdata = ExperimentData.from_numpy(domain=self.domain, input_array=x)
        x_experimentdata.run(data_generator.run)

        _, y = x_experimentdata.to_numpy()

        # Update the strategy based on fitness
        self.state = self.algorithm.tell(x, y.ravel(), state, self.evosax_param)

        # return the data
        return ExperimentData.from_numpy(domain=self.domain,
                                         input_array=x,
                                         output_array=y)


@dataclass
class EvoSaxCMAES_Parameters(OptimizerParameters):
    """Hyperparameters for EvoSaxCMAES optimizer"""

    population: int = 30


class EvoSaxCMAES(EvoSaxOptimizer):
    hyperparameters: EvoSaxCMAES_Parameters = EvoSaxCMAES_Parameters()
    evosax_algorithm = CMA_ES


class NeverGradOptimizer(NewOptimizer):

    @staticmethod
    def _check_imports():
        ...

    def update_step(self, data_generator: DataGenerator) -> ExperimentData:
        x = [self.algorithm.ask() for _ in range(self.hyperparameters.population)]

        # Evaluate the candidates
        x_experimentdata = ExperimentData.from_numpy(domain=self.domain, input_array=np.vstack([x_.value for x_ in x]))
        x_experimentdata.run(data_generator.run)

        _, y = x_experimentdata.to_numpy()
        for x_tell, y_tell in zip(x, y):
            self.algorithm.tell(x_tell, y_tell)

        # return the data
        return ExperimentData.from_numpy(domain=self.domain,
                                         input_array=np.vstack([x_.value for x_ in x]),
                                         output_array=y)


@dataclass
class DifferentialEvolution_Nevergrad_Parameters(OptimizerParameters):
    population: int = 30
    initialization: str = 'parametrization'
    scale: float = 1.0
    recommendation: str = 'optimistic'
    crossover: float = 0.5
    F1: float = 0.8
    F2: float = 0.8


class DifferentialEvolution_Nevergrad(NeverGradOptimizer):

    hyperparameters: DifferentialEvolution_Nevergrad_Parameters = DifferentialEvolution_Nevergrad_Parameters()

    def set_algorithm(self):
        p = ng.p.Array(shape=(len(self.domain),),
                       lower=self.domain.get_bounds()[:, 0], upper=self.domain.get_bounds()[:, 1])
        self.algorithm = ng.optimizers.DifferentialEvolution(initialization=self.hyperparameters.initialization,
                                                             popsize=self.hyperparameters.population,
                                                             scale=self.hyperparameters.scale,
                                                             recommendation=self.hyperparameters.recommendation,
                                                             crossover=self.hyperparameters.crossover,
                                                             F1=self.hyperparameters.F1,
                                                             F2=self.hyperparameters.F2)(p, budget=1e8)


@dataclass
class RandomSearch_Parameters(OptimizerParameters):
    """Hyperparameters for RandomSearch optimizer"""

    pass


class RandomSearch(NewOptimizer):
    """Naive random search"""

    hyperparameters: RandomSearch_Parameters = RandomSearch_Parameters()

    def update_step(self, data_generator: DataGenerator) -> ExperimentData:

        x_new = np.atleast_2d(
            [
                np.random.uniform(
                    low=self.domain.get_bounds()[d, 0], high=self.domain.get_bounds()[d, 1])
                for d in range(len(self.domain))
            ]
        )

        x_experimentdata = ExperimentData.from_numpy(domain=self.domain, input_array=x_new)

        # Evaluate the candidates
        x_experimentdata = ExperimentData.from_numpy(domain=self.domain, input_array=x_new)
        x_experimentdata.run(data_generator.run)

        _, y = x_experimentdata.to_numpy()

        # return the data
        return ExperimentData.from_numpy(domain=self.domain,
                                         input_array=x_new,
                                         output_array=y)

    def get_info(self) -> List[str]:
        return ['Fast', 'Single-Solution']


class _SciPyOptimizer(NewOptimizer):
    type: str = 'scipy'

    def _callback(self, xk: np.ndarray, *args, **kwargs) -> None:
        self.data += ExperimentSample.from_numpy(xk)

    def update_step(self):
        """Update step function"""
        raise ValueError(
            'Scipy optimizers don\'t have an update steps. Multiple iterations \
                 are directly called througout scipy.minimize.')

    def run_algorithm(self, iterations: int, data_generator: DataGenerator):
        """Run the algorithm for a number of iterations

        Parameters
        ----------
        iterations
            number of iterations
        function
            function to be evaluated
        """

        def fun(x):
            sample: ExperimentSample = data_generator.run(
                ExperimentSample.from_numpy(x))
            _, y = sample.to_numpy()
            return float(y)

        minimize(
            fun=fun,
            method=self.method,
            # TODO: #89 Fix this with the newest gradient method!
            jac='3-point',
            x0=self.data.get_n_best_input_parameters_numpy(
                nosamples=1).ravel(),
            callback=self._callback,
            options=self.hyperparameters.__dict__,
            bounds=self.domain.get_bounds(),
            tol=0.0,
        )


@dataclass
class LBFGSB_Parameters(OptimizerParameters):
    """Hyperparameters for LBFGSB optimizer"""

    ftol: float = 0.0
    gtol: float = 0.0


class LBFGSB(_SciPyOptimizer):
    """L-BFGS-B"""

    method: str = "L-BFGS-B"
    hyperparameters: LBFGSB_Parameters = LBFGSB_Parameters()

    def get_info(self) -> List[str]:
        return ['Stable', 'First-Order', 'Single-Solution']
