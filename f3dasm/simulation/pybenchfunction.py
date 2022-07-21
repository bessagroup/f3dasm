"""
Created by Axel Thevenot (2020)
Github repository: https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective
"""

import numpy as np
from f3dasm.base.simulation import Function


class PyBenchFunction(Function):
    @classmethod
    def is_dim_compatible(cls, d) -> bool:
        pass

    def scale_input(self, x: np.ndarray) -> np.ndarray:
        return (
            self.input_domain[:, 1] - self.input_domain[:, 0]
        ) * x + self.input_domain[:, 0]

    def descale_input(self, x: np.ndarray) -> np.ndarray:
        return (x - self.input_domain[:, 0]) / (
            self.input_domain[:, 1] - self.input_domain[:, 0]
        )

    def f(self, x: np.ndarray):
        if self.is_dim_compatible(self.dimensionality):
            return np.apply_along_axis(self, axis=1, arr=x).reshape(-1, 1)


class Thevenot(PyBenchFunction):
    name = "Thevenot"
    latex_formula = r"f(\mathbf{x}) = exp(-\sum_{i=1}^{d}(x_i / \beta)^{2m}) - 2exp(-\prod_{i=1}^{d}x_i^2) \prod_{i=1}^{d}cos^ 2(x_i) "
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-2\pi, 2\pi], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0)=-1, \text{ for}, m=5, \beta=15"
    continuous = True
    convex = True
    separable = True
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self, m=5, beta=15):
        d = self.dimensionality
        self.input_domain = np.array([[-2 * np.pi, 2 * np.pi] for _ in range(d)])
        self.m = m
        self.beta = beta

    def get_param(self):
        return {"m": self.m, "beta": self.beta}

    def get_global_minimum(self, d):
        X = np.array([0 for i in range(1, d + 1)])
        return (self.scale_input(X), self.eval(X))

    def __call__(self, X):
        d = X.shape[0]
        res = np.exp(-np.sum((X / self.beta) ** (2 * self.m)))
        res = res - 2 * np.exp(-np.prod(X**2)) * np.prod(np.cos(X) ** 2)
        return res


class Ackley(PyBenchFunction):
    name = "Ackley"
    latex_formula = r"f(\mathbf{x}) = -a \cdot exp(-b\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2})-exp(\frac{1}{d}\sum_{i=1}^{d}cos(c \cdot x_i))+ a + exp(1)"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-32, 32], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f((0, ..., 0)) = 0"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self, a=20, b=0.2, c=2 * np.pi):
        d = self.dimensionality
        self.input_domain = np.array([[-32, 32] for _ in range(d)])
        self.a = a
        self.b = b
        self.c = c

    def get_param(self):
        return {"a": self.a, "b": self.b, "c": self.c}

    def get_global_minimum(self, d):
        X = np.array([1 / (i + 1) for i in range(d)])
        X = np.array([0 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = -self.a * np.exp(-self.b * np.sqrt(np.mean(X**2)))
        res = res - np.exp(np.mean(np.cos(self.c * X))) + self.a + np.exp(1)
        return res


class AckleyN2(PyBenchFunction):
    name = "Ackley N. 2"
    latex_formula = r"f(x, y) = -200exp(-0.2\sqrt{x^2 + y^2)}"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-32, 32], y \in [-32, 32]"
    latex_formula_global_minimum = r"f(0, 0)=-200"
    continuous = False
    convex = True
    separable = False
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self=None):
        self.input_domain = np.array([[-32, 32], [-32, 32]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = -200 * np.exp(-0.2 * np.sqrt(x**2 + y**2))
        return res


class AckleyN3(PyBenchFunction):
    name = "Ackley N. 3"
    latex_formula = r"f(x, y) = -200exp(-0.2\sqrt{x^2 + y^2}) + 5exp(cos(3x) + sin(3y))"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-32, 32], y \in [-32, 32]"
    latex_formula_global_minimum = r"f(x, y)\approx-195.629028238419, at$$ $$x=\pm0.682584587365898, and$$ $$ y=-0.36075325513719"
    continuous = False
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self=None):
        self.input_domain = np.array([[-32, 32], [-32, 32]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0.682584587365898, -0.36075325513719])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = -200 * np.exp(-0.2 * np.sqrt(x**2 + y**2))
        res += 5 * np.exp(np.cos(3 * x) + np.sin(3 * y))
        return res


class AckleyN4(PyBenchFunction):
    name = "Ackley N. 4"
    latex_formula = r"f(\mathbf{x})=\sum_{i=1}^{d-1}\left( e^{-0.2}\sqrt{x_i^2+x_{i+1}^2} + 3\left( cos(2x_i) + sin(2x_{i+1}) \right) \right)"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-35, 35], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = (
        r"f(x, y)\approx-4.590101633799122, at$$ $$x=-1.51, and$$ $$ y=-0.755"
    )
    continuous = False
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self=None):
        d = self.dimensionality
        self.input_domain = np.array([[-35, 35] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        print("WARNING ! Is only is available for d=2")
        X = np.array([-1.51, -0.755])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        X, Xp1 = X[:-1], X[1]
        res = np.sum(
            np.exp(-0.2) * np.sqrt(X**2 + Xp1**2)
            + 3 * np.cos(2 * X)
            + np.sin(2 * Xp1)
        )
        return res


class Adjiman(PyBenchFunction):
    name = "Adjiman"
    latex_formula = r"f(x, y)=cos(x)sin(y) - \frac{x}{y^2+1}"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-1, 2], y \in [-1, 1]"
    latex_formula_global_minimum = r"f(0, 0)=-2.02181"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-1, 2], [-1, 1]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([1 / (i + 1) for i in range(d)])
        X = np.array([0, 0])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        x, y = X
        res = np.cos(x) * np.sin(y) - x / (y**2 + 1)
        return res


class AlpineN1(PyBenchFunction):
    name = "Alpine N. 1"
    latex_formula = r"f(\mathbf x) = \sum_{i=1}^{d}|x_i sin(x_i)+0.1x_i|"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [0, 10], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0)=0"
    continuous = False
    convex = False
    separable = True
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[0, 10] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for i in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        res = np.sum(np.abs(X * np.sin(X) + 0.1 * X))
        return res


class AlpineN2(PyBenchFunction):
    name = "Alpine N. 2"
    latex_formula = r"f(\mathbf x)=- \prod_{i=1}^{d}\sqrt{x_i}sin(x_i)"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [0, 10], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(7.917, ..., 7.917)=-2.808^d"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[0, 10] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([7.917 for i in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        res = -np.prod(np.sqrt(X) * np.sin(X))
        return res


class Bartels(PyBenchFunction):
    name = "Bartels"
    latex_formula = r"f(x,y)=|x^2 + y^2 + xy| + |sin(x)| + |cos(y)|"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-500, 500], y \in [-500, 500]"
    latex_formula_global_minimum = r"f(0, 0)=1"
    continuous = False
    convex = False
    separable = False
    differentiable = False
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-500, 500], [-500, 500]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = np.abs(x**2 + y**2 + x * y) + np.abs(np.sin(x)) + np.abs(np.cos(y))
        return res


class Beale(PyBenchFunction):
    name = "Beale"
    latex_formula = r"f(x, y) = (1.5-x+xy)^2+(2.25-x+xy^2)^2+(2.625-x+xy^3)^2"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-4.5, 4.5], y \in [-4.5, 4.5]"
    latex_formula_global_minimum = r"f(3, 0.5)=0"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-4.5, 4.5], [-4.5, 4.5]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([3, 0.5])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = (
            (1.5 - x + x * y) ** 2
            + (2.25 - x + x * y**2) ** 2
            + (2.625 - x + x * y**3) * 2
        )
        return res


class Bird(PyBenchFunction):
    name = "Bird"
    latex_formula = (
        r"f(x, y) = sin(x)exp((1-cos(y))^2)\\+cos(y)exp((1-sin(x))^2)+(x-y)^2"
    )
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-2\pi, 2\pi], y \in [-2\pi, 2\pi]"
    latex_formula_global_minimum = r"f(x, y)\approx-106.764537, at$$ $$(x, y)=(4.70104,3.15294), and$$ $$(x, y)=(-1.58214,-3.13024)"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([4.70104, 3.15294], [-1.58214, -3.13024])
        return (X, [self(x) for x in X])

    def __call__(self, X):
        x, y = X
        res = np.sin(x) * np.exp((1 - np.cos(y)) ** 2)
        res = res + np.cos(y) * np.exp((1 - np.sin(x)) ** 2) + (x - y) ** 2
        return res


class BohachevskyN1(PyBenchFunction):
    name = "Bohachevsky N. 1"
    latex_formula = r"f(x, y) = x^2 + 2y^2 -0.3cos(3\pi x)-0.4cos(4\pi y)+0.7"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-100, 100], y \in [-100, 100]"
    latex_formula_global_minimum = r"f(0, 0)=0"
    continuous = True
    convex = True
    separable = True
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-100, 100], [-100, 100]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = (
            x**2
            + 2 * y**2
            - 0.3 * np.cos(3 * np.pi * x)
            - 0.4 * np.cos(4 * np.pi * y)
            + 0.7
        )
        return res


class BohachevskyN2(PyBenchFunction):
    name = "Bohachevsky N. 2"
    latex_formula = r"f(x, y)=x^2 + 2y^2 -0.3cos(3\pi x)cos(4\pi y)+0.3"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-100, 100], y \in [-100, 100]"
    latex_formula_global_minimum = r"f(0, 0)=0"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-100, 100], [-100, 100]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = (
            x**2
            + 2 * y**2
            - 0.3 * np.cos(3 * np.pi * x) * np.cos(4 * np.pi * y)
            + 0.3
        )
        return res


class BohachevskyN3(PyBenchFunction):
    name = "Bohachevsky N. 3"
    latex_formula = r"f(x, y)=x^2 + 2y^2 -0.3cos(3\pi x + 4\pi y)cos(4\pi y)+0.3"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-50, 50], y \in [-50, 50]"
    latex_formula_global_minimum = r"f(0, 0)=0"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-50, 50], [-50, 50]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = (
            x**2
            + 2 * y**2
            - 0.3 * np.cos(3 * np.pi * x + 4 * np.pi * y) * np.cos(4 * np.pi * y)
            + 0.3
        )
        return res


class Booth(PyBenchFunction):
    name = "Booth"
    latex_formula = r"f(x,y)=(x+2y-7)^2+(2x+y-5)^2"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-10, 10], y \in [-10, 10]"
    latex_formula_global_minimum = r"f(1, 3)=0"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10], [-10, 10]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([1, 3])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
        return res


class Branin(PyBenchFunction):
    name = "Branin"
    latex_formula = r"f(x,y)=a(y - bx^2 + cx - r)^2 + s(1 - t)cos(x) + s"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-5, 10], y \in [0, 15]"
    latex_formula_global_minimum = r"f(x, y)\approx0.397887, at $$ $$(x, y)=(-\pi, 12.275),$$ $$(x, y)=(\pi, 2.275), and $$ $$(x, y)=(9.42478, 2.475) "
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(
        self,
        a=1,
        b=5.1 / (4 * np.pi**2),
        c=5 / np.pi,
        r=6,
        s=10,
        t=1 / (8 * np.pi),
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-5, 10], [0, 15]])
        self.a = a
        self.b = b
        self.c = c
        self.r = r
        self.s = s
        self.t = t

    def get_param(self):
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "r": self.r,
            "s": self.s,
            "t": self.t,
        }

    def get_global_minimum(self, d):
        X = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
        return (X, [self(x) for x in X])

    def __call__(self, X):
        x, y = X
        res = self.a * (y - self.b * x**2 + self.c * x - self.r) ** 2
        res = res + self.s * (1 - self.t) * np.cos(x) + self.s
        return res


class Brent(PyBenchFunction):
    name = "Brent"
    latex_formula = r"f(x, y) = (x + 10)^2 + (y + 10)^2 + exp(-x^2 - y^2)"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-20, 0], y \in [-20, 0]"
    latex_formula_global_minimum = r"f(-10, -10)=e^{-200}"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-20, 0], [-20, 0]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([-10, -10])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = (x + 10) ** 2 + (y + 10) ** 2 + np.exp(-(x**2) - y**2)
        return res


class Brown(PyBenchFunction):
    name = "Brown"
    latex_formula = r"f(\mathbf{x}) = \sum_{i=1}^{d-1}(x_i^2)^{(x_{i+1}^{2}+1)}+(x_{i+1}^2)^{(x_{i}^{2}+1)}"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-1, 4], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0)=0"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self=None):
        d = self.dimensionality
        self.input_domain = np.array([[-1, 4] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        X, Xp1 = X[:-1], X[1]
        res = np.sum((X**2) ** (Xp1**2 + 1) + (Xp1**2) ** (X**2 + 1))
        return res


class BukinN6(PyBenchFunction):
    name = "Bukin N. 6"
    latex_formula = r"f(x,y)=100\sqrt{|y-0.01x^2|}+0.01|x+10|"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-15, -5], y \in [-3, 3]"
    latex_formula_global_minimum = r"f(-10, 1)=0"
    continuous = True
    convex = True
    separable = False
    differentiable = False
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-15, -5], [-3, 3]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([-10, 1])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)
        return res


class Colville(PyBenchFunction):
    name = "Colville"
    latex_formula = r"f(\mathbf x) = 100(x_{1}^2-x_{2})^2 + (x_{1}-1)^2 + (x_{3} -1)^2  + 90(x_{3}^2-x_{4})^2\\+10.1((x_{2} - 1)^2+ (x_{4}-1)^2) +19.8(x_{2} -1)(x_{4} -1)"
    latex_formula_dimension = r"d=4"
    latex_formula_input_domain = (
        r"x_i \in [-10, 10], \forall i \in \llbracket 1, 4\rrbracket"
    )
    latex_formula_global_minimum = r"f(1, 1, 1, 1)=0"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 4

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10], [-10, 10], [-10, 10], [-10, 10]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([1, 1, 1, 1])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x1, x2, x3, x4 = X
        res = 100 * (x1**2 - x2) ** 2 + (x1 - 1) ** 2 + (x3 - 1) ** 2
        res = (
            res
            + 90 * (x3**2 - x4) ** 2
            + 10.1 * ((x2 - 1) ** 2 + (x4 - 1) ** 2)
            + 19.8 * (x2 - 1) * (x4 - 1)
        )
        return res


class CrossInTray(PyBenchFunction):
    name = "Cross-in-Tray"
    latex_formula = (
        r"f(x,y)=-0.0001(|sin(x)sin(y)exp(|100-\frac{\sqrt{x^2+y^2}}{\pi}|)|+1)^{0.1}"
    )
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-10, -10], y \in [-10, -10]"
    latex_formula_global_minimum = r"f(x, y)\approx-2.06261218, at $$ $$x\pm1.349406685353340, and $$ $$y\pm1.349406608602084"
    continuous = True
    convex = False
    separable = False
    differentiable = False
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10], [-10, 10]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array(
            [
                [-1.349406685353340, +1.349406608602084],
                [-1.349406685353340, -1.349406608602084],
                [+1.349406685353340, +1.349406608602084],
                [+1.349406685353340, -1.349406608602084],
            ]
        )
        return (X, [self(x) for x in X])

    def __call__(self, X):
        x, y = X
        res = (
            -0.0001
            * (
                np.abs(np.sin(x) * np.sin(y))
                * np.exp(np.abs(100 - np.sqrt(x**2 + y**2) / np.pi))
                + 1
            )
            ** 0.1
        )
        return res


class DeJongN5(PyBenchFunction):
    name = "De Jong N. 5"
    latex_formula = r"f(x, y)= \left ( 0.002 + \sum_{i=1}^{25} \frac{1}{i+(x - a_{1, i})^6+(x - a_{2, ij})^6}  \right)^{-1}"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-65.536, 65.536], y \in [-65.536, 65.536]"
    latex_formula_global_minimum = r"f(-32, -32)\approx-0.998003838818649"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self, a=None):
        d = self.dimensionality
        self.input_domain = np.array([[-65.536, 65.536], [-65.536, 65.536]])
        if a is None:
            l = [-32, -16, 0, 16, 32]
            self.a = np.array([[x, y] for x in l for y in l])
        else:
            self.a = a

    def get_param(self):
        return {"a": self.a}

    def get_global_minimum(self, d):
        X = self.a[0]
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = (
            0.002
            + np.sum(
                [
                    1 / ((i + 1) + (x - a1) ** 6 + (y - a2) ** 6)
                    for i, (a1, a2) in enumerate(self.a)
                ]
            )
        ) ** -1
        return res


class DeckkersAarts(PyBenchFunction):
    name = "Deckkers-Aarts"
    latex_formula = r"f(x, y) = 10^5x^2 + y^2 -(x^2 + y^2)^2 + 10^{-5}(x^2 + y^2)^4"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-20, 20], y \in [-20, 20]"
    latex_formula_global_minimum = r"f(0, \pm15)\approx25628.906250000004"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-20, 20], [-20, 20]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([[0, -15], [0, 15]])
        return (X, [self(x) for x in X])

    def __call__(self, X):
        x, y = X
        res = 1e5 * x**2 + y**2 - (x**2 + y**2) + 1e-5 * (x**2 + y**2) ** 4
        return res


class DixonPrice(PyBenchFunction):
    name = "Dixon Price"
    latex_formula = r"f(\mathbf x) = (x_1 - 1)^2 + \sum_{i=2}^d i(2x_i^2 - x_{i-1})^2"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = (
        r"f(\mathbf x) = 0\text{, at }x_i = 2^{-\frac{2^i-2}{2^i}}"
    )
    continuous = True
    convex = True
    separable = False
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([2 ** ((-(2 ** (i)) - 2) / 2**i) for i in range(1, d + 1)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = (X[0] - 1) ** 2 + np.sum(
            [(i + 1) * (2 * X[i] ** 2 - X[i - 1]) ** 2 for i in range(1, d)]
        )
        return res


class DropWave(PyBenchFunction):
    name = "Drop-Wave"
    latex_formula = (
        r"f(x, y) = - \frac{1 + cos(12\sqrt{x^{2} + y^{2}})}{(0.5(x^{2} + y^{2}) + 2)}"
    )
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-5.2, 5.2], y \in [-5.2, 5.2]"
    latex_formula_global_minimum = r"f(0, 0)=-1"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-5.2, 5.2], [-5.2, 5.2]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = -(1 + np.cos(12 * np.sqrt(x**2 + y**2))) / (
            0.5 * (x**2 + y**2) + 2
        )
        return res


class Easom(PyBenchFunction):
    name = "Easom"
    latex_formula = r"f(x,y)=-cos(x)cos(y) exp(-(x - \pi)^2-(y - \pi)^2)"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-100, 100], y \in [-100, 100]"
    latex_formula_global_minimum = r"f(\pi, \pi)=-1"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-100, 100], [-100, 100]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([np.pi, np.pi])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2) - (y - np.pi) ** 2)
        return res


class EggCrate(PyBenchFunction):
    name = "Egg Crate"
    latex_formula = r"f(x,y)=x^2 + y^2 + 25(sin^2(x) + sin^2(y))"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-5, 5], y \in [-5, 5]"
    latex_formula_global_minimum = r"f(0, 0)=-1"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-5, 5], [-5, 5]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = x**2 + y**2 + 25 * (np.sin(x) ** 2 + np.sin(y) ** 2)
        return res


class EggHolder(PyBenchFunction):
    name = "Egg Holder"
    latex_formula = r"f(x, y) = -(y + 47) sin \left ( \sqrt{| y + 0.5y +47 |} \right ) - x sin \left ( \sqrt{| x - (y + 47)|} \right )"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-512, 512], y \in [-512, 512]"
    latex_formula_global_minimum = r"f(512, 404.2319)=-1"
    continuous = False
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-512, 512], [-512, 512]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([512, 404.2319])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = -(y + 47) * np.sin(np.sqrt(np.abs(y + x / 2 + 47))) - x * np.sin(
            np.sqrt(np.abs(x - y - 47))
        )
        return res


class Exponential(PyBenchFunction):
    name = "Exponential"
    latex_formula = r"f(\mathbf{x})=-exp(-0.5\sum_{i=1}^d{x_i^2})"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-1, 1], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0) = -1"
    continuous = True
    convex = True
    separable = True
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-1, 1] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = -np.exp(-0.5 * np.sum(X**2))
        return res


class Forrester(PyBenchFunction):
    name = "Forrester"
    latex_formula = r"f(x)=(6x-2)^2sin(12x-4)"
    latex_formula_dimension = r"d=1"
    latex_formula_input_domain = r"x \in [0, 1]"
    latex_formula_global_minimum = r"f(0.757249) \approx -6.02074"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 1

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[0, 1] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0.757249])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x = X[0]
        res = ((6 * x - 2) ** 2) * np.sin(12 * x - 4)
        return res


class GoldsteinPrice(PyBenchFunction):
    name = "Goldstein-Price"
    latex_formula = r"f(x,y)=[1 + (x + y + 1)^2(19 - 14x+3x^2- 14y + 6xy + 3y^2)]\\ \cdot [30 + (2x - 3y)^2(18 - 32x + 12x^2 + 4y - 36xy + 27y^2)]"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-2, 2], y \in [-2, 2]"
    latex_formula_global_minimum = r"f(0, -1)=3"
    continuous = False
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-2, 2], [-2, 2]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0, -1])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = 1 + (x + y + 1) ** 2 * (
            19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2
        )
        res *= 30 + (2 * x - 3 * y) ** 2 * (
            18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2
        )
        return res


class GramacyLee(PyBenchFunction):
    name = "Gramacy & Lee"
    latex_formula = r"f(x) = \frac{sin(10\pi x)}{2x} + (x-1)^4"
    latex_formula_dimension = r"d=1"
    latex_formula_input_domain = r"x \in [-0.5, 2.5]"
    latex_formula_global_minimum = r"f(0.548563444114526) \approx -0.869011134989500"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 1

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-0.5, 2.5] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0.548563444114526])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x = X[0]
        res = np.sin(10 * np.pi * x) / 2 / x + (x - 1) ** 4
        return res


class Griewank(PyBenchFunction):
    name = "Griewank"
    latex_formula = r"f(\mathbf{x}) = 1 + \sum_{i=1}^{d} \frac{x_i^{2}}{4000} - \prod_{i=1}^{d}cos(\frac{x_i}{\sqrt{i}})"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-600, 600], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0) = 0"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-600, 600] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        i = np.arange(1, d + 1)
        res = 1 + np.sum(X**2 / 4000) - np.prod(np.cos(X / np.sqrt(i)))
        return res


class HappyCat(PyBenchFunction):
    name = "Happy Cat"
    latex_formula = r"f(\mathbf{x})=\left[\left(||\mathbf{x}||^2 - d\right)^2\right]^\alpha + \frac{1}{d}\left(\frac{1}{2}||\mathbf{x}||^2+\sum_{i=1}^{d}x_i\right)+\frac{1}{2}"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-2, 2], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(-1, ..., -1) = 0"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self, alpha=0.5):
        d = self.dimensionality
        self.input_domain = np.array([[-2, 2] for _ in range(d)])
        self.alpha = alpha

    def get_param(self):
        return {"alpha": self.alpha}

    def get_global_minimum(self, d):
        X = np.array([-1 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        norm = np.sum(X**2)
        res = ((norm - d) ** 2) ** self.alpha + (1 / d) * (0.5 * norm + np.sum(X)) + 0.5
        return res


class Himmelblau(PyBenchFunction):
    name = "Himmelblau"
    latex_formula = r"f(x, y) = (x^{2} + y - 11)^{2} + (x + y^{2} - 7)^{2}"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-6, 6], y \in [-6, 6]"
    latex_formula_global_minimum = r"f(3,2)=0$$$$f(-2.805118,3.283186)\approx0$$$$f(-3.779310,-3.283186)\approx0$$$$f(3.584458,-1.848126)\approx0"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-6, 6], [-6, 6]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array(
            [
                [3, 2],
                [-2.805118, 3.283186],
                [-3.779310, -3.283186],
                [3.584458, -1.848126],
            ]
        )
        return (X, [self(x) for x in X])

    def __call__(self, X):
        x, y = X
        res = (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2
        return res


class HolderTable(PyBenchFunction):
    name = "Holder-Table"
    latex_formula = r"f(x,y)=-|sin(x)cos(y)exp(|1-\frac{\sqrt{x^2+y^2}}{\pi}|)|"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-10, 10], y \in [-10, 10]"
    latex_formula_global_minimum = r"f(\pm8.05502, \pm9.66459)\approx-19.2085"
    continuous = True
    convex = False
    separable = False
    differentiable = False
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10], [-10, 10]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array(
            [
                [-8.05502, 9.66459],
                [-8.05502, -9.66459],
                [8.05502, 9.66459],
                [8.05502, -9.66459],
            ]
        )
        return (X, [self(x) for x in X])

    def __call__(self, X):
        x, y = X
        res = -np.abs(
            np.sin(x) * np.cos(y) * np.exp(np.abs(1 - np.sqrt(x**2 + y**2) / np.pi))
        )
        return res


class Keane(PyBenchFunction):
    name = "Keane"
    latex_formula = r"f(x,y)=-\frac{\sin^2(x-y)\sin^2(x+y)}{\sqrt{x^2+y^2}}"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-10, 10], y \in [-10, 10]"
    latex_formula_global_minimum = r"f(1.393249070031784,0)\approx0.673667521146855 $$$$f(0,1.393249070031784)\approx0.673667521146855"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10], [-10, 10]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([[1.393249070031784, 0], [0, 1.393249070031784]])
        return (X, [self(x) for x in X])

    def __call__(self, X):
        x, y = X
        res = -(np.sin(x - y) ** 2 * np.sin(x + y) ** 2) / np.sqrt(x**2 + y**2)
        return res


class Langermann(PyBenchFunction):
    name = "Langermann"
    latex_formula = r"f(\mathbf x) = \sum_{i=1}^{m}c_iexp\left( -\frac{1}{\pi}\sum_{j=1}^{d}(x_j - A_{ij})^2\right)cos\left( \pi\sum_{j=1}^{d}(x_j - A_{ij})^2\right)"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [0, 10], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"\text{Not found...}"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self, m=None, c=None, A=None):
        d = self.dimensionality
        self.input_domain = np.array([[0, 10] for _ in range(d)])
        self.m = m if m is not None else 5
        self.c = c if c is not None else np.array([1, 2, 5, 2, 3])
        self.A = (
            A if A is not None else np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])
        )

    def get_param(self):
        return {"m": self.m, "c": self.c, "A": self.A}

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = np.sum(
            [
                self.c[i]
                * np.exp(-1 / np.pi * np.sum((X - self.A[i]) ** 2))
                * np.cos(np.pi * np.sum((X - self.A[i]) ** 2))
                for i in range(self.m)
            ]
        )
        return res


class Leon(PyBenchFunction):
    name = "Leon"
    latex_formula = r"f(x, y) = 100(y - x^{3})^2 + (1 - x)^2"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [0, 10], y \in [0, 10]"
    latex_formula_global_minimum = r"f(1, 1)=0"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[0, 10], [0, 10]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([1, 1])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = 100 * (y - x**3) ** 2 + (1 - x) ** 2
        return res


class LevyN13(PyBenchFunction):
    name = "Levy N. 13"
    latex_formula = (
        r"f(x, y) = sin^2(3\pi x)+(x-1)^2(1+sin^2(3\pi y))+(y-1)^2(1+sin^2(2\pi y))"
    )
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-10, 10], y \in [-10, 10]"
    latex_formula_global_minimum = r"f(1, 1)=0"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10], [-10, 10]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([1, 1])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = (
            np.sin(3 * np.pi * x) ** 2
            + (x - 1) ** 2 * (1 + np.sin(3 * np.pi * y) ** 2)
            + (y - 1) ** 2 * (1 + np.sin(2 * np.pi * y) ** 2)
        )
        return res


class Matyas(PyBenchFunction):
    name = "Matyas"
    latex_formula = r"f(x, y)=0.26(x^2+y^2) -0.48xy"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-10, 10], y \in [-10, 10]"
    latex_formula_global_minimum = r"f(0, 0)=0"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10], [-10, 10]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = 0.26 * (x**2 + y**2) - 0.48 * x * y
        return res


class McCormick(PyBenchFunction):
    name = "McCormick"
    latex_formula = r"f(x, y)=sin(x + y) + (x - y) ^2 - 1.5x + 2.5 y + 1"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-1.5, 4], y \in [-3, 3]"
    latex_formula_global_minimum = r"f(-0.547,-1.547)\approx=-1.9133"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-1.5, 4], [-3, 3]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([-0.547, -1.547])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1
        return res


class Michalewicz(PyBenchFunction):
    name = "Michalewicz"
    latex_formula = r"f(\mathbf x) = - \sum_{i=1}^{d}sin(x_i)sin\left(\frac{ix_i^2}{\pi}\right)^{2m}"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [0, \pi], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"\text{at $d=2$, }f(2.20, 1.57) \approx -1.8013"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self, m=10):
        d = self.dimensionality
        self.input_domain = np.array([[0, np.pi] for _ in range(d)])
        self.m = m

    def get_param(self):
        return {"m": self.m}

    def get_global_minimum(self, d):
        assert d == 2, "Michalewicz minimum is only given for d=2"
        X = np.array([2.20, 1.57])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        i = np.arange(1, d + 1)
        res = -np.sum(np.sin(X) * np.sin(i * X**2 / np.pi) ** (2 * self.m))
        return res


class Periodic(PyBenchFunction):
    name = "Periodic"
    latex_formula = (
        r"f(\mathbf{x})= 1 + \sum_{i=1}^{d}{sin^2(x_i)}-0.1exp(-\sum_{i=1}^{d}x_i^2)"
    )
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"\text{at $d=2$, }f(0, ..., 0) =0.9"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = 1 + np.sum(np.sin(X) ** 2) - 0.1 * np.exp(-np.sum(X**2))
        return res


class PermZeroDBeta(PyBenchFunction):
    name = "Perm 0, d, beta"
    latex_formula = r"f(\bold{x}) = \sum_{i=1}^{d} \left ( \sum_{j=1}^{d}(j + \beta) \left ( x_{j}^{i} - \frac{1}{j^{i}}\right )  \right )^2"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-d, d], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f((1, \frac{1}{2}, ..., \frac{1}{d})) = 0"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self, beta=10):
        d = self.dimensionality
        self.input_domain = np.array([[-d, d] for _ in range(d)])
        self.beta = beta

    def get_param(self):
        return {"beta": self.beta}

    def get_global_minimum(self, d):
        X = np.array([1 / (i + 1) for i in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = np.sum(
            [
                (
                    np.sum(
                        [
                            ((j + 1) + self.beta * (X[j] ** (i + 1) - j ** (i + 1)))
                            for j in range(d)
                        ]
                    )
                )
                ** 2
                for i in range(d)
            ]
        )
        return res


class PermDBeta(PyBenchFunction):
    name = "Perm d, beta"
    latex_formula = r"f(\bold{x}) = \sum_{i=1}^{d} \left ( \sum_{j=1}^{d}(j^i + \beta) \left( \left( \frac{x_{j}}{j}\right )^{i} - 1 \right )  \right )^2"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-d, d], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(1, 2, ..., d) = 0"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self, beta=0.5):
        d = self.dimensionality
        self.input_domain = np.array([[-d, d] for _ in range(d)])
        self.beta = beta

    def get_param(self):
        return {"beta": self.beta}

    def get_global_minimum(self, d):
        X = np.array([1 / (i + 1) for i in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        j = np.arange(1, d + 1)
        res = np.sum(
            [
                np.sum((j**i + self.beta) * ((X / j) ** i - 1)) ** 2
                for i in range(1, d + 1)
            ]
        )
        return res


class Powell(PyBenchFunction):
    name = "Powell"
    latex_formula = r"f(\mathbf{x})=\sum_{i=1}^{d}|x_i|^{i+1}"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-1, 1], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0) =0"
    continuous = True
    convex = True
    separable = True
    differentiable = False
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-1, 1] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = np.sum(np.abs(X) ** np.arange(2, d + 2))
        return res


class Qing(PyBenchFunction):
    name = "Qing"
    latex_formula = r"f(\mathbf{x})=\sum_{i=1}^{d}(x^2-i)^2"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-500, 500], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(\pmi, ..., \pmi) =0"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-500, 500] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        X = np.array([range(d)]) + 1
        for i in range(d):
            neg = X.copy()
            neg[:, i] *= -1
            X = np.vstack((X, neg))
        return (X, [self(x) for x in X])

    def __call__(self, X):
        d = X.shape[0]
        X1 = np.power(X, 2)

        res = 0
        for i in range(d):
            res = res + np.power(X1[i] - (i + 1), 2)
        return res


class Quartic(PyBenchFunction):
    name = "Quartic"
    latex_formula = r"f(\mathbf{x})=\sum_{i=1}^{n}ix_i^4+\text{random}[0,1)"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-1.28, 1.28], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0) = 0 + \text{random noise}"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    mutimodal = True
    randomized_term = True
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-1.28, 1.28] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = np.sum(np.arange(1, d + 1) * X**4) + np.random.random()
        return res


class Rastrigin(PyBenchFunction):
    name = "Rastrigin"
    latex_formula = r"f(\mathbf{x})=\sum_{i=1}^{d}(x_i^2 - 10cos(2\pi x_i))"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-5.12, 5.12], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0) = 0"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-5.12, 5.12] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = 10 * d + np.sum(X**2 - 10 * np.cos(2 * np.pi * X))
        return res


class Ridge(PyBenchFunction):
    name = "Ridge"
    latex_formula = r"f(\mathbf{x})=x_1 + \beta\left(\sum_{i=2}^{d}x_i^2\right)^\alpha"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-5, 5], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"\text{On the hypercube } [-\gamma, \gamma]^d,$$$$ f(-\gamma, 0, 0..., 0)=-\gamma "
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self, beta=2, alpha=0.1):
        d = self.dimensionality
        self.input_domain = np.array([[-5, 5] for _ in range(d)])
        self.beta = beta
        self.alpha = alpha

    def get_param(self):
        return {"beta": self.beta, "alpha": self.alpha}

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        X[0] = self.input_domain[0, 0]
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = X[0] + self.beta * np.sum(X[1:] ** 2) ** self.alpha
        return res


class Rosenbrock(PyBenchFunction):
    name = "Rosenbrock"
    latex_formula = (
        r"f(\mathbf{x})=\sum_{i=1}^{d-1}[b (x_{i+1} - x_i^2)^ 2 + (a - x_i)^2]"
    )
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-5, 10], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(1, ..., 1)=0"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self, a=1, b=100):
        d = self.dimensionality
        self.input_domain = np.array([[-5, 10] for _ in range(d)])
        self.a = a
        self.b = b

    def get_param(self):
        return {"a": self.a, "b": self.b}

    def get_global_minimum(self, d):
        X = np.array([1 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = np.sum(
            np.abs(self.b * (X[1:] - X[:-1] ** 2) ** 2 + (self.a - X[:-1]) ** 2)
        )
        return res


class RotatedHyperEllipsoid(PyBenchFunction):
    name = "Rotated Hyper-Ellipsoid"
    latex_formula = r"f(\mathbf{x})=\sum_{i=1}^{d}\sum_{j=1}^{i}x_j^2"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-65.536, 65.536], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0)=0"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-65.536, 65.536] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = np.sum([np.sum(X[: i + 1] ** 2) for i in range(d)])
        return res


class Salomon(PyBenchFunction):
    name = "Salomon"
    latex_formula = r"f(\mathbf{x})=1-cos(2\pi\sqrt{\sum_{i=1}^{d}x_i^2})+0.1\sqrt{\sum_{i=1}^{d}x_i^2}"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-100, 100], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0)=0"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = 1 - np.cos(2 * np.pi * np.sqrt(np.sum(X**2)))
        res = res + 0.1 * np.sqrt(np.sum(X**2))
        return res


class SchaffelN1(PyBenchFunction):
    name = "Schaffel N. 1"
    latex_formula = r"f(x, y)=0.5 + \frac{sin^2(x^2+y^2)^2-0.5}{(1+0.001(x^2+y^2))^2}"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-100, 100], y \in [-100, 100]"
    latex_formula_global_minimum = r"f(0, 0)=0"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-100, 100], [-100, 100]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = (
            0.5
            + (np.sin((x**2 + y**2) ** 2) ** 2 - 0.5)
            / (1 + 0.001 * (x**2 + y**2)) ** 2
        )
        return res


class SchaffelN2(PyBenchFunction):
    name = "Schaffel N. 2"
    latex_formula = r"f(x, y)=0.5 + \frac{sin^2(x^2-y^2)-0.5}{(1+0.001(x^2+y^2))^2}"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-100, 100], y \in [-100, 100]"
    latex_formula_global_minimum = r"f(0, 0)=0"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-4, 4], [-4, 4]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = (
            0.5
            + (np.sin((x**2 + y**2)) ** 2 - 0.5)
            / (1 + 0.001 * (x**2 + y**2)) ** 2
        )
        return res


class SchaffelN3(PyBenchFunction):
    name = "Schaffel N. 3"
    latex_formula = (
        r"f(x, y)=0.5 + \frac{sin^2(cos(|x^2-y^2|))-0.5}{(1+0.001(x^2+y^2))^2}"
    )
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-100, 100], y \in [-100, 100]"
    latex_formula_global_minimum = r"f(0,1.253115)\approx0.00156685"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-4, 4], [-4, 4]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0, 1.253115])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = (
            0.5
            + (np.sin(np.cos(np.abs(x**2 + y**2))) ** 2 - 0.5)
            / (1 + 0.001 * (x**2 + y**2)) ** 2
        )
        return res


class SchaffelN4(PyBenchFunction):
    name = "Schaffel N. 4"
    latex_formula = (
        r"f(x, y)=0.5 + \frac{sin^2(cos(|x^2-y^2|))-0.5}{(1+0.001(x^2+y^2))^2}"
    )
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-100, 100], y \in [-100, 100]"
    latex_formula_global_minimum = r"f(0, 1.253115)\approx0.292579"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-4, 4], [-4, 4]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0, 1.253115])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = (
            0.5
            + (np.cos(np.sin(np.abs(x**2 + y**2))) ** 2 - 0.5)
            / (1 + 0.001 * (x**2 + y**2)) ** 2
        )
        return res


class Schwefel(PyBenchFunction):
    name = "Schwefel"
    latex_formula = r"f(\mathbf{x})=418.9829d -{\sum_{i=1}^{d} x_i sin(\sqrt{|x_i|})}"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-500, 500], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(420.9687, ..., 420.9687)=0"
    continuous = True
    convex = False
    separable = True
    differentiable = False
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-500, 500] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([420.9687 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = 418.9829 * d - np.sum(X * np.sin(np.sqrt(np.abs(X))))
        return res


class Schwefel2_20(PyBenchFunction):
    name = "Schwefel 2.20"
    latex_formula = r"f(\mathbf{x})=\sum_{i=1}^d |x_i|"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-100, 100], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0)=0"
    continuous = True
    convex = True
    separable = True
    differentiable = False
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = np.sum(np.abs(X))
        return res


class Schwefel2_21(PyBenchFunction):
    name = "Schwefel 2.21"
    latex_formula = r"f(\mathbf{x})=\max_{i \in \llbracket 1, d\rrbracket}|x_i| "
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-100, 100], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0)=0"
    continuous = True
    convex = True
    separable = True
    differentiable = False
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = np.max(np.abs(X))
        return res


class Schwefel2_22(PyBenchFunction):
    name = "Schwefel 2.22"
    latex_formula = r"f(\mathbf{x})=\sum_{i=1}^{d}|x_i|+\prod_{i=1}^{d}|x_i|"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-100, 100], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0)=0"
    continuous = True
    convex = True
    separable = True
    differentiable = False
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = np.sum(np.abs(X)) + np.prod(np.abs(X))
        return res


class Schwefel2_23(PyBenchFunction):
    name = "Schwefel 2.23"
    latex_formula = r"f(\mathbf{x})=\sum_{i=1}^{d}x_i^{10}"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0)=0"
    continuous = True
    convex = True
    separable = True
    differentiable = False
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = np.sum(X**10)
        return res


class Shekel(PyBenchFunction):
    name = "Shekel"
    latex_formula = r"f(\mathbf x) = -\sum_{i=1}^{m}\left(\sum_{j=1}^{4} (x_j - C_{ij})^2 + \beta_i\right)^{-1}"
    latex_formula_dimension = r"d=4"
    latex_formula_input_domain = (
        r"x_i \in [0, 10], \forall i \in \llbracket 1, 4\rrbracket"
    )
    latex_formula_global_minimum = (
        r"f(4, 4, 4, 4)= -10.1532 \text{ with default params}"
    )
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 4

    def set_parameters(self, m=None, C=None, beta=None):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10], [-10, 10], [-10, 10], [-10, 10]])
        self.m = m if m is not None else 10
        self.beta = (
            beta
            if beta is not None
            else 1 / 10 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
        )
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

    def get_param(self):
        return {"m": self.m, "C": self.C, "beta": self.beta}

    def get_global_minimum(self, d):
        X = self.C[0]
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x1, x2, x3, x4 = X
        res = -np.sum(
            [
                np.sum((X[i] - self.C[i]) ** 2 + self.beta[i]) ** -1
                for i in range(self.m)
            ]
        )
        return res


class Shubert(PyBenchFunction):
    name = "Shubert"
    latex_formula = (
        r"f(\mathbf{x})=\prod_{i=1}^{d}{\left(\sum_{j=1}^5{ cos((j+1)x_i+j)}\right)}"
    )
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = (
        r"\text{ has 18 global minima }f(\mathbf{x*})\approx-186.7309"
    )
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        return None

    def __call__(self, X):
        d = X.shape[0]
        for i in range(0, d):
            res = np.prod(
                np.sum([i * np.cos((j + 1) * X[i] + j) for j in range(1, 5 + 1)])
            )
        return res


class ShubertN3(PyBenchFunction):
    name = "Shubert N. 3"
    latex_formula = r"f(\mathbf{x})=\sum_{i=1}^{d}{\sum_{j=1}^5{j sin((j+1)x_i+j)}}"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(-7.4, -7.4)\approx-29.673336786222684"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([-7.4, -7.4])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = np.sum(np.sum([j * np.sin((j + 1) * X + j) for j in range(1, 5 + 1)]))
        return res


class ShubertN4(PyBenchFunction):
    name = "Shubert N. 4"
    latex_formula = r"f(\mathbf{x})=\sum_{i=1}^{d}{\sum_{j=1}^5{j cos((j+1)x_i+j)}}"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(4.85, 4.85)\approx-25.720968549936323"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([4.85, 4.85])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = np.sum(np.sum([j * np.cos((j + 1) * X + j) for j in range(1, 5 + 1)]))
        return res


class Sphere(PyBenchFunction):
    name = "Sphere"
    latex_formula = r"f(\mathbf{x})=\sum_{i=1}^{d} x_i^{2}"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-5.12, 5.12], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0)=0"
    continuous = True
    convex = True
    separable = True
    differentiable = False
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-5.12, 5.12] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = np.sum(X**2)
        return res


class StyblinskiTank(PyBenchFunction):
    name = "Styblinski Tank"
    latex_formula = r"f(\mathbf{x})=\frac{1}{2}\sum_{i=1}^{d} (x_i^4 -16x_i^2+5x_i)"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-5, 5], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(-2.903534, ..., -2.903534)=-39.16599d"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(
        self,
    ):
        d = self.dimensionality
        self.input_domain = np.array([[-5, 5] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([-2.903534 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = 0.5 * np.sum(X**4 - 16 * X**2 + 5 * X)
        return res


class SumSquares(PyBenchFunction):
    name = "Sum Squares"
    latex_formula = r"f(\mathbf{x})=\sum_{i=1}^{d}ix_i^{2}"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0) =0"
    continuous = True
    convex = True
    separable = True
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        i = np.arange(1, d + 1)
        res = np.sum(i * X**2)
        return res


class ThreeHump(PyBenchFunction):
    name = "Three-Hump"
    latex_formula = r"f(x,y)=2x^2-1.05x^4+\frac{x^6}{6}+xy+y^2"
    latex_formula_dimension = r"d=2"
    latex_formula_input_domain = r"x \in [-5, 5], y \in [-5, 5]"
    latex_formula_global_minimum = r"f(0, 0)=0"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-5, 5], [-5, 5]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0, 0])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y = X
        res = 2 * x**2 - 1.05 * x**4 + x**6 * (1 / 6) + x * y + y**2
        return res


class Trid(PyBenchFunction):
    name = "Trid"
    latex_formula = (
        r"f(\mathbf{x})=\sum_{i=1}^{d}(x_i-1)^2-\sum_{i=2}^{d}(x_i-1x_{i-1})"
    )
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-d^2, d^2], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = (
        r"f(\mathbf{x}) =\frac{-d(d+4)(d-1)}{6}, $$$$x_i=i(d+1-i)"
    )
    continuous = True
    convex = True
    separable = False
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-(d**2), d**2] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([i * (d + 1 - i) for i in range(1, d + 1)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        i = np.arange(1, d + 1)
        res = np.sum(X - 1) ** 2 - np.sum(X[1:] * X[:-1])
        return res


class Wolfe(PyBenchFunction):
    name = "Wolfe"
    latex_formula = r"f(x, y, z) = \frac{4}{3}(x^2 + y^2 - xy)^{0.75} + z"
    latex_formula_dimension = r"d=3"
    latex_formula_input_domain = (
        r"x_i \in [0, 2], \forall i \in \llbracket 1, 3\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, 0, 0)=0"
    continuous = True
    convex = False
    separable = False
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 3

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[0, 2], [0, 2], [0, 2]])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0, 0, 0])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        x, y, z = X
        res = 4 / 3 * (x**2 + y**2 - x * y) ** 0.75 + z
        return res


class XinSheYang(PyBenchFunction):
    name = "Xin She Yang"
    latex_formula = r"f(\mathbf{x})=\sum_{i=1}^{d}\text{random}_i[0,1)|x_i|^i"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-5, 5], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0)=0 \text{ (without random)}"
    continuous = False
    convex = False
    separable = True
    differentiable = False
    mutimodal = True
    randomized_term = True
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-5, 5] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for i in range(1, d + 1)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        i = np.arange(1, d + 1)
        rand = np.random.random(d)
        res = np.sum(rand * np.abs(X) ** i)
        return res


class XinSheYangN2(PyBenchFunction):
    name = "Xin She Yang N.2"
    latex_formula = r"f(\mathbf{x})=(\sum_{i=1}^{d}|x_i|)exp(-\sum_{i=1}^{d}sin(x_i^2))"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-2\pi, 2\pi], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0)=0"
    continuous = False
    convex = False
    separable = False
    differentiable = False
    mutimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-2 * np.pi, 2 * np.pi] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for i in range(1, d + 1)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = np.sum(np.abs(X)) * np.exp(-np.sum(np.sin(X**2)))
        return res


class XinSheYangN3(PyBenchFunction):
    name = "Xin She Yang N.3"
    latex_formula = r"f(\mathbf{x})=exp(-\sum_{i=1}^{n}(x_i / \beta)^{2m}) - 2exp(-\sum_{i=1}^{n}x_i^2) \prod_{i=1}^{n}cos^ 2(x_i)"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-2\pi, 2\pi], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0)=-1, \text{ for}, m=5, \beta=15"
    continuous = True
    convex = True
    separable = False
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = True

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self, m=5, beta=15):
        d = self.dimensionality
        self.input_domain = np.array([[-2 * np.pi, 2 * np.pi] for _ in range(d)])
        self.m = m
        self.beta = beta

    def get_param(self):
        return {"m": self.m, "beta": self.beta}

    def get_global_minimum(self, d):
        X = np.array([0 for i in range(1, d + 1)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = np.exp(-np.sum((X / self.beta) ** (2 * self.m)))
        res = res - 2 * np.exp(-np.sum(X**2)) * np.prod(np.cos(X) ** 2)
        return res


class XinSheYangN4(PyBenchFunction):
    name = "Xin-She Yang N.4"
    latex_formula = r"f(\mathbf{x})=\left(\sum_{i=1}^{d}sin^2(x_i)-exp(-\sum_{i=1}^{d}x_i^2)\right)exp(-\sum_{i=1}^{d}{sin^2\sqrt{|x_i|}})"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0)=-2"
    continuous = True
    convex = True
    separable = False
    differentiable = False
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-10, 10] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for i in range(1, d + 1)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        res = np.sum(np.sin(X) ** 2 - np.exp(-np.sum(X) ** 2)) * np.exp(
            -np.sum(np.sin(np.sqrt(np.abs(X))) ** 2)
        )
        return res


class Zakharov(PyBenchFunction):
    name = "Zakharov"
    latex_formula = r"f(\mathbf{x})=\sum_{i=1}^n x_i^{2}+(\sum_{i=1}^n 0.5ix_i)^2 + (\sum_{i=1}^n 0.5ix_i)^4"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-5, 10], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0)=-1"
    continuous = False
    convex = False
    separable = False
    differentiable = False
    mutimodal = False
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[-5, 10] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([0 for i in range(1, d + 1)])
        return (self.descale_input(X), self.eval(self.descale_input(X)))

    def __call__(self, X):
        d = X.shape[0]
        i = np.arange(1, d + 1)
        res = np.sum(X**2) + np.sum(0.5 * i * X) ** 2 + np.sum(0.5 * i * X) ** 4
        return res
