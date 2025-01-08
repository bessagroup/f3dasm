"""
Created by Axel Thevenot (2020)
Github repository:
 https://github.com/AxelThevenot/Python_Benchmark_Test_
Optimization_Function_Single_Objective
"""
#                                                                       Modules
# =============================================================================

# Third-party
import autograd.numpy as np

# Locals
from ..functions.adapters.pybenchfunction import PyBenchFunction

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class Thevenot(PyBenchFunction):
    """.. image:: ../img/functions/Thevenot.png"""

    name = "Thevenot"
    continuous = True
    convex = True
    separable = True
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self, m=5, beta=15):
        d = self.dimensionality
        self.input_domain = np.array(
            [[-2 * np.pi, 2 * np.pi] for _ in range(d)])
        self.m = m
        self.beta = beta

    def get_global_minimum(self, d):
        X = np.array([0 for i in range(1, d + 1)])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        res = np.exp(-np.sum((X / self.beta) ** (2 * self.m)))
        res = res - 2 * np.exp(-np.prod(X**2)) * np.prod(np.cos(X) ** 2)
        return res


class Ackley(PyBenchFunction):
    """.. image:: ../img/functions/Ackley.png"""

    name = "Ackley"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self, a=20, b=0.2, c=2 * np.pi):
        d = self.dimensionality
        self.input_domain = np.array([[-32.768, 32.768] for _ in range(d)])
        self.a = a
        self.b = b
        self.c = c

    def get_global_minimum(self, d):
        X = np.array([1 / (i + 1) for i in range(d)])
        X = np.array([0 for _ in range(d)])
        return (
            self._retrieve_original_input(X), self(
                self._retrieve_original_input(X)))

    def evaluate(self, X):
        res = -self.a * np.exp(-self.b * np.sqrt(np.mean(X**2)))
        res = res - np.exp(np.mean(np.cos(self.c * X))) + self.a + np.exp(1)
        return res


class AckleyN2(PyBenchFunction):
    """.. image:: ../img/functions/AckleyN2.png"""

    name = "Ackley N. 2"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False
    error_autograd = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self=None):
        self.input_domain = np.array([[-32, 32], [-32, 32]])

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = -200 * np.exp(-0.2 * np.sqrt(x**2 + y**2))
        return res


class AckleyN3(PyBenchFunction):
    """.. image:: ../img/functions/AckleyN3.png"""

    name = "Ackley N. 3"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self=None):
        self.input_domain = np.array([[-32, 32], [-32, 32]])

    def get_global_minimum(self, d):
        X = np.array([0.682584587365898, -0.36075325513719])
        Y = np.array([[-195.629028238419]])
        return (self._retrieve_original_input(X), Y)

    def evaluate(self, X):
        x, y = X
        res = -200 * np.exp(-0.2 * np.sqrt(x**2 + y**2))
        res += 5 * np.exp(np.cos(3 * x) + np.sin(3 * y))
        return res


class AckleyN4(PyBenchFunction):
    """.. image:: ../img/functions/AckleyN4.png"""

    name = "Ackley N. 4"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 1)

    def _set_parameters(self=None):
        d = self.dimensionality
        self.input_domain = np.array([[-35, 35] for _ in range(d)])

    def get_global_minimum(self, d):

        if d != 2:  # WARNING ! Is only is available for d=2
            # This is the global minimum for d=2
            return (None, np.array([[-4.5901016]]))

        X = np.array([-1.51, -0.755])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        X, Xp1 = X[:-1], X[1]
        res = np.sum(np.exp(-0.2) * np.sqrt(
            X**2 + Xp1**2) + 3 * np.cos(2 * X) + np.sin(2 * Xp1))
        return res


class Adjiman(PyBenchFunction):
    """.. image:: ../img/functions/Adjiman.png"""

    name = "Adjiman"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-1, 2], [-1, 1]])

    def get_global_minimum(self, d):
        X = np.array([1 / (i + 1) for i in range(d)])
        X = np.array([0, 0])
        Y = np.array([[-2.02180678]])
        return (self._retrieve_original_input(X), Y)

    def evaluate(self, X):
        x, y = X
        res = np.cos(x) * np.sin(y) - x / (y**2 + 1)
        return res


class Bartels(PyBenchFunction):
    """.. image:: ../img/functions/Bartels.png"""

    name = "Bartels"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-500, 500], [-500, 500]])

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = np.abs(x**2 + y**2 + x * y) + \
            np.abs(np.sin(x)) + np.abs(np.cos(y))
        return res


class Beale(PyBenchFunction):
    """.. image:: ../img/functions/Beale.png"""

    name = "Beale"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-4.5, 4.5], [-4.5, 4.5]])

    def get_global_minimum(self, d):
        X = np.array([3, 0.5])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = (
            1.5 - x + x * y) ** 2 + (
                2.25 - x + x * y**2) ** 2 + (2.625 - x + x * y**3) ** 2
        return res


class Bird(PyBenchFunction):
    """.. image:: ../img/functions/Bird.png"""

    name = "Bird"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array(
            [[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])

    def get_global_minimum(self, d):
        X = np.array([[4.70104, 3.15294], [-1.58214, -3.13024]])
        return (self._retrieve_original_input(X),
                [self(x) for x in self._retrieve_original_input(X)])

    def evaluate(self, X):
        x, y = X
        res = np.sin(x) * np.exp((1 - np.cos(y)) ** 2)
        res = res + np.cos(y) * np.exp((1 - np.sin(x)) ** 2) + (x - y) ** 2
        return res


class BohachevskyN1(PyBenchFunction):
    """.. image:: ../img/functions/BohachevskyN1.png"""

    name = "Bohachevsky N. 1"
    continuous = True
    convex = True
    separable = True
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-100, 100], [-100, 100]])

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = x**2 + 2 * y**2 - 0.3 * \
            np.cos(3 * np.pi * x) - 0.4 * np.cos(4 * np.pi * y) + 0.7
        return res


class BohachevskyN2(PyBenchFunction):
    """.. image:: ../img/functions/BohachevskyN2.png"""

    name = "Bohachevsky N. 2"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-100, 100], [-100, 100]])

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = x**2 + 2 * y**2 - 0.3 * \
            np.cos(3 * np.pi * x) * np.cos(4 * np.pi * y) + 0.3
        return res


class BohachevskyN3(PyBenchFunction):
    """.. image:: ../img/functions/BohachevskyN3.png"""

    name = "Bohachevsky N. 3"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-50, 50], [-50, 50]])

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = x**2 + 2 * y**2 - 0.3 * \
            np.cos(3 * np.pi * x + 4 * np.pi * y) * np.cos(4 * np.pi * y) + 0.3
        return res


class Booth(PyBenchFunction):
    """.. image:: ../img/functions/Booth.png"""

    name = "Booth"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-10, 10], [-10, 10]])

    def get_global_minimum(self, d):
        X = np.array([1, 3])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
        return res


class Branin(PyBenchFunction):
    """.. image:: ../img/functions/Branin.png"""

    name = "Branin"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(
        self,
        a=1,
        b=5.1 / (4 * np.pi**2),
        c=5 / np.pi,
        r=6,
        s=10,
        t=1 / (8 * np.pi),
    ):
        self.input_domain = np.array([[-5, 10], [0, 15]])
        self.a = a
        self.b = b
        self.c = c
        self.r = r
        self.s = s
        self.t = t

    def get_global_minimum(self, d):
        X = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
        return (self._retrieve_original_input(X),
                [self(x) for x in self._retrieve_original_input(X)])

    def evaluate(self, X):
        x, y = X
        res = self.a * (y - self.b * x**2 + self.c * x - self.r) ** 2
        res = res + self.s * (1 - self.t) * np.cos(x) + self.s
        return res


class Brent(PyBenchFunction):
    """.. image:: ../img/functions/Brent.png"""

    name = "Brent"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-20, 0], [-20, 0]])

    def get_global_minimum(self, d):
        X = np.array([-10, -10])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = (x + 10) ** 2 + (y + 10) ** 2 + np.exp(-(x**2) - y**2)
        return res


class Brown(PyBenchFunction):
    """.. image:: ../img/functions/Brown.png"""

    name = "Brown"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 1)

    def _set_parameters(self=None):
        d = self.dimensionality
        self.input_domain = np.array([[-1, 4] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        X, Xp1 = X[:-1], X[1]
        res = np.sum((X**2) ** (Xp1**2 + 1) + (Xp1**2) ** (X**2 + 1))
        return res


class BukinN6(PyBenchFunction):
    """.. image:: ../img/functions/BukinN6.png"""

    name = "Bukin N. 6"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False
    error_autograd = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-15, -5], [-3, 3]])

    def get_global_minimum(self, d):
        X = np.array([-10, 1])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)
        return res


class Colville(PyBenchFunction):
    """.. image:: ../img/functions/Colville.png"""

    name = "Colville"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 4

    def _set_parameters(self):
        self.input_domain = np.array(
            [[-10, 10], [-10, 10], [-10, 10], [-10, 10]])

    def get_global_minimum(self, d):
        X = np.array([1, 1, 1, 1])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        x1, x2, x3, x4 = X
        res = 100 * (x1**2 - x2) ** 2 + (x1 - 1) ** 2 + (x3 - 1) ** 2
        res = res + 90 * (x3**2 - x4) ** 2 + 10.1 * (
            (x2 - 1)
            ** 2 + (x4 - 1) ** 2) + 19.8 * (x2 - 1) * (x4 - 1)
        return res


class CrossInTray(PyBenchFunction):
    """.. image:: ../img/functions/CrossInTray.png"""

    name = "Cross-in-Tray"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-10, 10], [-10, 10]])

    def get_global_minimum(self, d):
        X = np.array(
            [
                [-1.349406685353340, +1.349406608602084],
                [-1.349406685353340, -1.349406608602084],
                [+1.349406685353340, +1.349406608602084],
                [+1.349406685353340, -1.349406608602084],
            ]
        )
        return (self._retrieve_original_input(X),
                [self(x) for x in self._retrieve_original_input(X)])

    def evaluate(self, X):
        x, y = X
        res = (
            -0.0001 * (np.abs(np.sin(x) * np.sin(y)) * np.exp(
                np.abs(100 - np.sqrt(x**2 + y**2) / np.pi)) + 1) ** 0.1
        )
        return res


class DeJongN5(PyBenchFunction):
    """.. image:: ../img/functions/DeJongN5.png"""

    name = "De Jong N. 5"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self, a=None):
        self.input_domain = np.array([[-65.536, 65.536], [-65.536, 65.536]])
        if a is None:
            l_parameter = [-32, -16, 0, 16, 32]
            self.a = np.array([[x, y]
                              for x in l_parameter for y in l_parameter])
        else:
            self.a = a

    def get_global_minimum(self, d):
        X = self.a[0]
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = (
            0.002 + np.sum([1 / ((i + 1) + (x - a1) ** 6 + (y - a2) ** 6)
                           for i, (a1, a2) in enumerate(self.a)])
        ) ** -1
        return res


class DeckkersAarts(PyBenchFunction):
    """.. image:: ../img/functions/DeckkersAarts.png"""

    name = "Deckkers-Aarts"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-20, 20], [-20, 20]])

    def get_global_minimum(self, d):
        X = np.array([[0, -15], [0, 15]])
        return (self._retrieve_original_input(X),
                [self(x) for x in self._retrieve_original_input(X)])

    def evaluate(self, X):
        x, y = X
        res = 1e5 * x**2 + y**2 - (x**2 + y**2) ** 2 + \
            1e-5 * (x**2 + y**2) ** 4
        return res


class DixonPrice(PyBenchFunction):
    """.. image:: ../img/functions/DixonPrice.png"""

    name = "Dixon Price"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([2 ** -(((2 ** (i)) - 2) / 2**i)
                     for i in range(1, d + 1)])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        d = X.shape[0]
        res = (X[0] - 1) ** 2 + np.sum(
            [(i + 1) * (2 * X[i] ** 2 - X[i - 1]) ** 2 for i in range(1, d)])
        return res


class DropWave(PyBenchFunction):
    """.. image:: ../img/functions/DropWave.png"""

    name = "Drop-Wave"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-5.2, 5.2], [-5.2, 5.2]])

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = -(1 + np.cos(12 * np.sqrt(x**2 + y**2))) / \
            (0.5 * (x**2 + y**2) + 2)
        return res


class Easom(PyBenchFunction):
    """.. image:: ../img/functions/Easom.png"""

    name = "Easom"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-100, 100], [-100, 100]])

    def get_global_minimum(self, d):
        X = np.array([np.pi, np.pi])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = -np.cos(x) * np.cos(y) * \
            np.exp(-((x - np.pi) ** 2) - (y - np.pi) ** 2)
        return res


class EggCrate(PyBenchFunction):
    """.. image:: ../img/functions/EggCrate.png"""

    name = "Egg Crate"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-5, 5], [-5, 5]])

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = x**2 + y**2 + 25 * (np.sin(x) ** 2 + np.sin(y) ** 2)
        return res


class EggHolder(PyBenchFunction):
    """.. image:: ../img/functions/EggHolder.png"""

    name = "Egg Holder"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-512, 512], [-512, 512]])

    def get_global_minimum(self, d):
        X = np.array([512, 404.2319])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = -(y + 47) * np.sin(np.sqrt(np.abs(y + 0.5 * x + 47))) - \
            x * np.sin(np.sqrt(np.abs(x - (y + 47))))
        return res


class Exponential(PyBenchFunction):
    """.. image:: ../img/functions/Exponential.png"""

    name = "Exponential"
    continuous = True
    convex = True
    separable = True
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-1, 1] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        res = -np.exp(-0.5 * np.sum(X**2))
        return res


class GoldsteinPrice(PyBenchFunction):
    """.. image:: ../img/functions/GoldsteinPrice.png"""

    name = "Goldstein-Price"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-2, 2], [-2, 2]])

    def get_global_minimum(self, d):
        X = np.array([0, -1])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = 1 + (
            x + y + 1) ** 2 * (
            19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
        res *= 30 + (2 * x - 3 * y) ** 2 * (
            18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
        return res


class Griewank(PyBenchFunction):
    """.. image:: ../img/functions/Griewank.png"""

    name = "Griewank"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-600, 600] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        d = X.shape[0]
        i = np.arange(1, d + 1)
        res = 1 + np.sum(X**2 / 4000) - np.prod(np.cos(X / np.sqrt(i)))
        return res


class HappyCat(PyBenchFunction):
    """.. image:: ../img/functions/HappyCat.png"""

    name = "Happy Cat"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = True
    error_autograd = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self, alpha=0.5):
        d = self.dimensionality
        self.input_domain = np.array([[-2, 2] for _ in range(d)])
        self.alpha = alpha

    def get_global_minimum(self, d):
        X = np.array([-1 for _ in range(d)])
        return (self._retrieve_original_input(X), self(
            self._retrieve_original_input(X)))

    def evaluate(self, X):
        d = X.shape[0]
        norm = np.sum(X**2)
        res = ((norm - d) ** 2) ** self.alpha + \
            (1 / d) * (0.5 * norm + np.sum(X)) + 0.5
        return res


class Himmelblau(PyBenchFunction):
    """.. image:: ../img/functions/Himmelblau.png"""

    name = "Himmelblau"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-6, 6], [-6, 6]])

    def get_global_minimum(self, d):
        X = np.array(
            [
                [3, 2],
                [-2.805118, 3.283186],
                [-3.779310, -3.283186],
                [3.584458, -1.848126],
            ]
        )
        return (self._retrieve_original_input(X),
                [self(x) for x in self._retrieve_original_input(X)])

    def evaluate(self, X):
        x, y = X
        res = (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2
        return res


class HolderTable(PyBenchFunction):
    """.. image:: ../img/functions/HolderTable.png"""

    name = "Holder-Table"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-10, 10], [-10, 10]])

    def get_global_minimum(self, d):
        X = np.array(
            [
                [-8.05502, 9.66459],
                [-8.05502, -9.66459],
                [8.05502, 9.66459],
                [8.05502, -9.66459],
            ]
        )
        return (self._retrieve_original_input(X),
                [self(x) for x in self._retrieve_original_input(X)])

    def evaluate(self, X):
        x, y = X
        res = -np.abs(np.sin(x) * np.cos(y) * np.exp(
            np.abs(1 - np.sqrt(x**2 + y**2) / np.pi)))
        return res


class Keane(PyBenchFunction):
    """.. image:: ../img/functions/Keane.png"""

    name = "Keane"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False
    error_autograd = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-10, 10], [-10, 10]])

    def get_global_minimum(self, d):
        X = np.array([[1.393249070031784, 0], [0, 1.393249070031784]])
        return (self._retrieve_original_input(X),
                [self(x) for x in self._retrieve_original_input(X)])

    def evaluate(self, X):
        x, y = X
        res = -(np.sin(x - y) ** 2 * np.sin(x + y) ** 2) / np.sqrt(x**2 + y**2)
        return res


class Langermann(PyBenchFunction):
    """.. image:: ../img/functions/Langermann.png"""

    name = "Langermann"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self, m=None, c=None, A=None):
        d = self.dimensionality
        self.input_domain = np.array([[0, 10] for _ in range(d)])
        self.m = m if m is not None else 5
        self.c = c if c is not None else np.array([1, 2, 5, 2, 3])
        self.A = A if A is not None else np.array(
            [[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        # Global minimum is not known but the following
        # is definitely smaller than the global minimum
        Y = np.array([[-4.5]])
        return (self._retrieve_original_input(X), Y)

    def evaluate(self, X):
        res = np.sum(
            [
                self.c[i] * np.exp(-1 / np.pi * np.sum(
                    (X - self.A[i]) ** 2)) * np.cos(
                        np.pi * np.sum((X - self.A[i]) ** 2))
                for i in range(self.m)
            ]
        )
        return res


class Leon(PyBenchFunction):
    """.. image:: ../img/functions/Leon.png"""

    name = "Leon"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[0, 10], [0, 10]])

    def get_global_minimum(self, d):
        X = np.array([1, 1])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = 100 * (y - x**3) ** 2 + (1 - x) ** 2
        return res


class Levy(PyBenchFunction):
    """.. image:: ../img/functions/Levy.png"""

    name = "Levy"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([1 for _ in range(d)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        z = 1 + (X - 1) / 4
        res = (
            np.sin(np.pi * z[0]) ** 2 + sum(
                (z[:-1] - 1) ** 2 * (1 + 10 * np.sin(
                    np.pi * z[:-1] + 1) ** 2)) + (
                        z[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * z[-1]) ** 2)
        )
        return res


class LevyN13(PyBenchFunction):
    """.. image:: ../img/functions/LevyN13.png"""

    name = "Levy N. 13"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-10, 10], [-10, 10]])

    def get_global_minimum(self, d):
        X = np.array([1, 1])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = (
            np.sin(3 * np.pi * x) ** 2 + (x - 1) ** 2 * (
                1 + np.sin(3 * np.pi * y) ** 2) + (y - 1) ** 2 * (
                    1 + np.sin(2 * np.pi * y) ** 2)
        )
        return res


class Matyas(PyBenchFunction):
    """.. image:: ../img/functions/Matyas.png"""

    name = "Matyas"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-10, 10], [-10, 10]])

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = 0.26 * (x**2 + y**2) - 0.48 * x * y
        return res


class McCormick(PyBenchFunction):
    """.. image:: ../img/functions/McCormick.png"""

    name = "McCormick"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-1.5, 4], [-3, 3]])

    def get_global_minimum(self, d):
        X = np.array([-0.547, -1.547])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1
        return res


class Michalewicz(PyBenchFunction):
    """.. image:: ../img/functions/Michalewicz.png"""

    name = "Michalewicz"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self, m=10):
        d = self.dimensionality
        self.input_domain = np.array([[0, np.pi] for _ in range(d)])
        self.m = m

    def get_global_minimum(self, d):
        if d != 2:  # Michalewicz minimum is only given for d=2
            # Calculated with polyfit
            Y = np.array(
                [[4.49903414e-04*(d**2) - 2.15704771e-01*d - 4.85292809e+00]])
            return (None, Y)  # Substituted minimum for d=2
        X = np.array([2.20, 1.57])
        Y = np.array([[-1.8013]])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        d = X.shape[0]
        i = np.arange(1, d + 1)
        res = -np.sum(np.sin(X) * np.sin(i * X**2 / np.pi) ** (2 * self.m))
        return res


class Periodic(PyBenchFunction):
    """.. image:: ../img/functions/Periodic.png"""

    name = "Periodic"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        res = 1 + np.sum(np.sin(X) ** 2) - 0.1 * np.exp(-np.sum(X**2))
        return res


class Powell(PyBenchFunction):
    """.. image:: ../img/functions/Powell.png"""

    name = "Powell"
    continuous = True
    convex = True
    separable = True
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-1, 1] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        d = X.shape[0]
        res = np.sum(np.abs(X) ** np.arange(2, d + 2))
        return res


class Qing(PyBenchFunction):
    """.. image:: ../img/functions/Qing.png"""

    name = "Qing"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-500, 500] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        X = np.array([range(d)]) + 1
        for i in range(d):
            neg = X.copy()
            neg[:, i] *= -1
            X = np.vstack((X, neg))
        return (self._retrieve_original_input(X),
                [self(x) for x in self._retrieve_original_input(X)])

    def evaluate(self, X):
        d = X.shape[0]
        X1 = np.power(X, 2)

        res = 0
        for i in range(d):
            res = res + np.power(X1[i] - (i + 1), 2)
        return res


class Quartic(PyBenchFunction):
    """.. image:: ../img/functions/Quartic.png"""

    name = "Quartic"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    multimodal = True
    randomized_term = True
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-1.28, 1.28] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        # Global minimum value without the randomized term
        Y = np.array([[0.0]])
        return (self._retrieve_original_input(X), Y)

    def evaluate(self, X):
        d = X.shape[0]
        res = np.sum(np.arange(1, d + 1) * X**4)
        return res


class Rastrigin(PyBenchFunction):
    """.. image:: ../img/functions/Rastrigin.png"""

    name = "Rastrigin"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-5.12, 5.12] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        d = X.shape[0]
        res = 10 * d + np.sum(X**2 - 10 * np.cos(2 * np.pi * X))
        return res


class Ridge(PyBenchFunction):
    """.. image:: ../img/functions/Ridge.png"""

    name = "Ridge"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = True
    error_autograd = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self, beta=2, alpha=0.1):
        d = self.dimensionality
        self.input_domain = np.array([[-5, 5] for _ in range(d)])
        self.beta = beta
        self.alpha = alpha

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        X[0] = self.input_domain[0, 0]
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        res = X[0] + self.beta * np.sum(X[1:] ** 2) ** self.alpha
        return res


class Rosenbrock(PyBenchFunction):
    """.. image:: ../img/functions/Rosenbrock.png"""

    name = "Rosenbrock"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self, a=1, b=100):
        d = self.dimensionality
        self.input_domain = np.array([[-5, 10] for _ in range(d)])
        self.a = a
        self.b = b

    def get_global_minimum(self, d):
        X = np.array([1 for _ in range(d)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        res = np.sum(
            np.abs(self.b * (X[1:] - X[:-1] ** 2) ** 2 + (
                self.a - X[:-1]) ** 2))
        return res


class RotatedHyperEllipsoid(PyBenchFunction):
    """.. image:: ../img/functions/RotatedHyperEllipsoid.png"""

    name = "Rotated Hyper-Ellipsoid"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-65.536, 65.536] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        d = X.shape[0]
        res = np.sum([np.sum(X[: i + 1] ** 2) for i in range(d)])
        return res


class Salomon(PyBenchFunction):
    """.. image:: ../img/functions/Salomon.png"""

    name = "Salomon"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        res = 1 - np.cos(2 * np.pi * np.sqrt(np.sum(X**2)))
        res = res + 0.1 * np.sqrt(np.sum(X**2))
        return res


class SchaffelN1(PyBenchFunction):
    """.. image:: ../img/functions/SchaffelN1.png"""

    name = "Schaffel N. 1"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-100, 100], [-100, 100]])

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = 0.5 + (np.sin((x**2 + y**2) ** 2) ** 2 - 0.5) / \
            (1 + 0.001 * (x**2 + y**2)) ** 2
        return res


class SchaffelN2(PyBenchFunction):
    """.. image:: ../img/functions/SchaffelN2.png"""

    name = "Schaffel N. 2"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-4, 4], [-4, 4]])

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = 0.5 + (np.sin((x**2 + y**2)) ** 2 - 0.5) / \
            (1 + 0.001 * (x**2 + y**2)) ** 2
        return res


class SchaffelN3(PyBenchFunction):
    """.. image:: ../img/functions/SchaffelN3.png"""

    name = "Schaffel N. 3"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-4, 4], [-4, 4]])

    def get_global_minimum(self, d):
        X = np.array([0, 1.253115])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = 0.5 + (np.sin(np.cos(
            np.abs(x**2 + y**2))) ** 2 - 0.5) / (
                1 + 0.001 * (x**2 + y**2)) ** 2
        return res


class SchaffelN4(PyBenchFunction):
    """.. image:: ../img/functions/SchaffelN4.png"""

    name = "Schaffel N. 4"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-4, 4], [-4, 4]])

    def get_global_minimum(self, d):
        X = np.array([0, 1.253115])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = 0.5 + (np.cos(np.sin(np.abs(
            x**2 + y**2))) ** 2 - 0.5) / (
                1 + 0.001 * (x**2 + y**2)) ** 2
        return res


class Schwefel(PyBenchFunction):
    """.. image:: ../img/functions/Schwefel.png"""

    name = "Schwefel"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-500, 500] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([420.9687 for _ in range(d)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        d = X.shape[0]
        res = 418.9829 * d - np.sum(X * np.sin(np.sqrt(np.abs(X))))
        return res


class Schwefel2_20(PyBenchFunction):
    """.. image:: ../img/functions/Schwefel2_20.png"""

    name = "Schwefel 2.20"
    continuous = True
    convex = True
    separable = True
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        res = np.sum(np.abs(X))
        return res


class Schwefel2_21(PyBenchFunction):
    """.. image:: ../img/functions/Schwefel2_21.png"""

    name = "Schwefel 2.21"
    continuous = True
    convex = True
    separable = True
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        res = np.max(np.abs(X))
        return res


class Schwefel2_22(PyBenchFunction):
    """.. image:: ../img/functions/Schwefel2_22.png"""

    name = "Schwefel 2.22"
    continuous = True
    convex = True
    separable = True
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False
    error_autograd = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        res = np.sum(np.abs(X)) + np.prod(np.abs(X))
        return res


class Schwefel2_23(PyBenchFunction):
    """.. image:: ../img/functions/Schwefel2_23.png"""

    name = "Schwefel 2.23"
    continuous = True
    convex = True
    separable = True
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        res = np.sum(X**10)
        return res


class Shekel(PyBenchFunction):
    """.. image:: ../img/functions/Shekel.png"""

    name = "Shekel"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 4

    def _set_parameters(self, m=None, C=None, beta=None):
        self.input_domain = np.array(
            [[-10, 10], [-10, 10], [-10, 10], [-10, 10]])
        self.m = m if m is not None else 10
        self.beta = beta if beta is not None else 1 / \
            10 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
        self.C = (
            C
            if C is not None
            else np.array(
                [
                    [4, 4, 4, 4],
                    [1, 1, 1, 1],
                    [8, 8, 8, 8],
                    [6, 6, 6, 6],
                    [3, 7, 3, 7],
                    [2, 9, 2, 9],
                    [5, 3, 5, 3],
                    [8, 1, 8, 1],
                    [6, 2, 6, 2],
                    [7, 3.6, 7, 3.6],
                ]
            )
        )

    def get_global_minimum(self, d):
        X = self.C[0]
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        x1, x2, x3, x4 = X
        res = -np.sum([[np.sum((X - self.C[i]) ** 2 + self.beta[i]) ** -1]
                      for i in range(self.m)])
        return res


class Shubert(PyBenchFunction):
    """.. image:: ../img/functions/Shubert.png"""

    name = "Shubert"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10] for _ in range(d)])

    def get_global_minimum(self, d):
        # Global minimum from
        # https://documentation.sas.com/doc/en/orcdc/14.2/
        # orlsoug/orlsoug_ga_gettingstarted09.htm"
        # X = np.array([-7.708309818, -0.800371886])
        # Has 18 global minima around -186.7309
        return (None, np.array([[-186.7309]]))

    def evaluate(self, X):
        d = X.shape[0]
        for i in range(0, d):
            res = np.prod(np.sum([i * np.cos((j + 1) * X[i] + j)
                          for j in range(1, 5 + 1)]))
        return res


class ShubertN3(PyBenchFunction):
    # """.. image:: ../img/functions/ShubertN3.png"""

    name = "Shubert N. 3"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([-7.4 for _ in range(d)])

        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        res = np.sum(np.sum([j * np.sin((j + 1) * X + j)
                     for j in range(1, 5 + 1)]))
        return res


class ShubertN4(PyBenchFunction):
    """.. image:: ../img/functions/ShubertN4.png"""

    name = "Shubert N. 4"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([4.85 for _ in range(d)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        res = np.sum(np.sum([j * np.cos((j + 1) * X + j)
                     for j in range(1, 5 + 1)]))
        return res


class Sphere(PyBenchFunction):
    """.. image:: ../img/functions/Sphere.png"""

    name = "Sphere"
    continuous = True
    convex = True
    separable = True
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-5.12, 5.12] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        res = np.sum(X**2)
        return res


class StyblinskiTang(PyBenchFunction):
    """.. image:: ../img/functions/StyblinskiTang.png"""

    name = "Styblinski Tang"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-5, 5] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([-2.903534 for _ in range(d)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        res = 0.5 * np.sum(X**4 - 16 * X**2 + 5 * X)
        return res


class SumSquares(PyBenchFunction):
    """.. image:: ../img/functions/SumSquares.png"""

    name = "Sum Squares"
    continuous = True
    convex = True
    separable = True
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        d = X.shape[0]
        i = np.arange(1, d + 1)
        res = np.sum(i * X**2)
        return res


class ThreeHump(PyBenchFunction):
    """.. image:: ../img/functions/ThreeHump.png"""

    name = "Three-Hump"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def _set_parameters(self):
        self.input_domain = np.array([[-5, 5], [-5, 5]])

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y = X
        res = 2 * x**2 - 1.05 * x**4 + x**6 * (1 / 6) + x * y + y**2
        return res


class Trid(PyBenchFunction):
    """.. image:: ../img/functions/Trid.png"""

    name = "Trid"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-(d**2), d**2] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([i * (d + 1 - i) for i in range(1, d + 1)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        res = np.sum((X - 1) ** 2) - np.sum(X[1:] * X[:-1])
        return res


class Wolfe(PyBenchFunction):
    """.. image:: ../img/functions/Wolfe.png"""

    name = "Wolfe"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 3

    def _set_parameters(self):
        self.input_domain = np.array([[0, 2], [0, 2], [0, 2]])

    def get_global_minimum(self, d):
        X = np.array([0, 0, 0])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        x, y, z = X
        res = (4 / 3) * (x**2 + y**2 - x * y) ** 0.75 + z
        return res


class XinSheYang(PyBenchFunction):
    """.. image:: ../img/functions/XinSheYang.png"""

    name = "Xin She Yang"
    continuous = True
    convex = False
    separable = True
    differentiable = False
    multimodal = True
    randomized_term = True
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-5, 5] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for i in range(1, d + 1)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        d = X.shape[0]
        i = np.arange(1, d + 1)
        res = np.sum(np.abs(X) ** i)
        return res


class XinSheYangN2(PyBenchFunction):
    """.. image:: ../img/functions/XinSheYangN2.png"""

    name = "Xin She Yang N.2"
    continuous = False
    convex = False
    separable = False
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False
    error_autograd = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array(
            [[-2 * np.pi, 2 * np.pi] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for i in range(1, d + 1)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        res = np.sum(np.abs(X)) * np.exp(-np.sum(np.sin(X**2)))
        return res


class XinSheYangN3(PyBenchFunction):
    """.. image:: ../img/functions/XinSheYangN3.png"""

    name = "Xin She Yang N.3"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self, m=5, beta=15):
        d = self.dimensionality
        self.input_domain = np.array(
            [[-2 * np.pi, 2 * np.pi] for _ in range(d)])
        self.m = m
        self.beta = beta

    def get_global_minimum(self, d):
        X = np.array([0 for i in range(1, d + 1)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        res = np.exp(-np.sum((X / self.beta) ** (2 * self.m)))
        res = res - 2 * np.exp(-np.sum(X**2)) * np.prod(np.cos(X) ** 2)
        return res


class XinSheYangN4(PyBenchFunction):
    """.. image:: ../img/functions/XinSheYangN4.png"""

    name = "Xin-She Yang N.4"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False
    error_autograd = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for i in range(1, d + 1)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        res = np.sum(np.sin(X) ** 2 - np.exp(-np.sum(X) ** 2)) * \
            np.exp(-np.sum(np.sin(np.sqrt(np.abs(X))) ** 2))
        return res


class Zakharov(PyBenchFunction):
    """.. image:: ../img/functions/Zakharov.png"""

    name = "Zakharov"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    multimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def _set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-5, 10] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for i in range(1, d + 1)])
        return (self._retrieve_original_input(X),
                self(self._retrieve_original_input(X)))

    def evaluate(self, X):
        d = X.shape[0]
        i = np.arange(1, d + 1)
        res = np.sum(X**2) + np.sum(0.5 * i * X) ** 2 + \
            np.sum(0.5 * i * X) ** 4
        return res


__all__ = ['Ackley',
           'AckleyN2',
           'AckleyN3',
           'AckleyN4',
           'Adjiman',
           'Bartels',
           'Beale',
           'Bird',
           'BohachevskyN1',
           'BohachevskyN2',
           'BohachevskyN3',
           'Booth',
           'Branin',
           'Brent',
           'Brown',
           'BukinN6',
           'Colville',
           'CrossInTray',
           'DeJongN5',
           'DeckkersAarts',
           'DixonPrice',
           'DropWave',
           'Easom',
           'EggCrate',
           'EggHolder',
           'Exponential',
           'GoldsteinPrice',
           'Griewank',
           'HappyCat',
           'Himmelblau',
           'HolderTable',
           'Keane',
           'Langermann',
           'Leon',
           'Levy',
           'LevyN13',
           'Matyas',
           'McCormick',
           'Michalewicz',
           'Periodic',
           'Powell',
           'Qing',
           'Quartic',
           'Rastrigin',
           'Ridge',
           'Rosenbrock',
           'RotatedHyperEllipsoid',
           'Salomon',
           'SchaffelN1',
           'SchaffelN2',
           'SchaffelN3',
           'SchaffelN4',
           'Schwefel',
           'Schwefel2_20',
           'Schwefel2_21',
           'Schwefel2_22',
           'Schwefel2_23',
           'Shekel',
           'Shubert',
           'ShubertN3',
           'ShubertN4',
           'Sphere',
           'StyblinskiTang',
           'SumSquares',
           'Thevenot',
           'ThreeHump',
           'Trid',
           'Wolfe',
           'XinSheYang',
           'XinSheYangN2',
           'XinSheYangN3',
           'XinSheYangN4',
           'Zakharov']
