#                                                                       Modules
# =============================================================================

# Third-party
import numpy as np

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def ackley(x: np.ndarray, a: float = 20., b: float = 0.2,
           c: float = 2 * np.pi):
    """Ackley function

    Parameters
    ----------
    x : np.ndarray
        Input array
    a : float, optional
        Parameter a, by default 20
    b : float, optional
        Parameter b, by default 0.2
    c : float, optional
        Parameter c, by default 2 * np.pi

    Returns
    -------
    float
        The value of the Ackley function at x.
    """
    y = -a * np.exp(-b * np.sqrt(np.mean(x**2)))
    y = y - np.exp(np.mean(np.cos(c * x))) + a + np.exp(1)
    return y


def beale(x: np.ndarray):
    """Beale function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Beale function at x.
    """
    y = (1.5 - x[0] + x[0] * x[1]) ** 2 + \
        (2.25 - x[0] + x[0] * x[1]**2) ** 2 + \
        (2.625 - x[0] + x[0] * x[1]**3) ** 2
    return y


def bohachevsky(x: np.ndarray):
    """Bohachevsky N. 1 function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Bohachevsky N. 1 function at x.
    """
    y = x[0]**2 + 2 * x[1]**2 - 0.3 * np.cos(3 * np.pi * x[0]) - \
        0.4 * np.cos(4 * np.pi * x[1]) + 0.7
    return y


def booth(x: np.ndarray):
    """Booth function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Booth function at x.
    """
    y = (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
    return y


def branin(x: np.ndarray, a=1, b=5.1 / (4 * np.pi**2), c=5 / np.pi,
           r=6, s=10, t=1 / (8 * np.pi)):
    """Branin function

    Parameters
    ----------
    x : np.ndarray
        Input array
    a : float, optional
        Parameter a, by default 1
    b : float, optional
        Parameter b, by default 5.1 / (4 * np.pi**2)
    c : float, optional
        Parameter c, by default 5 / np.pi
    r : float, optional
        Parameter r, by default 6
    s : float, optional
        Parameter s, by default 10
    t : float, optional
        Parameter t, by default 1 / (8 * np.pi)

    Returns
    -------
    float
        The value of the Branin function at x.
    """
    y = a * (x[1] - b * x[0]**2 + c * x[0] - r) ** 2
    y = y + s * (1 - t) * np.cos(x[0]) + s
    return y


def bukin(x: np.ndarray):
    """Bukin N. 6 function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Bukin N. 6 function at x.
    """
    y = 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0]**2)) + 0.01 * np.abs(x[0] + 10)
    return y


def crossintray(x: np.ndarray):
    """Cross-in-Tray function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Cross-in-Tray function at x.
    """
    y = -0.0001 * (np.abs(np.sin(x[0]) * np.sin(x[1])) * np.exp(
        np.abs(100 - np.sqrt(x[0]**2 + x[1]**2) / np.pi)) + 1) ** 0.1
    return y


def dixonprice(x: np.ndarray):
    """Dixon Price function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Dixon Price function at x.
    """
    d = x.shape[0]
    y = (x[0] - 1) ** 2 + np.sum(
        [(i + 1) * (2 * x[i] ** 2 - x[i - 1]) ** 2 for i in range(1, d)])
    return y


def drop_wave(x: np.ndarray):
    """Drop-Wave function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Drop-Wave function at x.
    """
    y = -(1 + np.cos(12 * np.sqrt(x[0]**2 + x[1]**2))) / \
        (0.5 * (x[0]**2 + x[1]**2) + 2)
    return y


def easom(x: np.ndarray):
    """Easom function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Easom function at x.
    """
    y = -np.cos(x[0]) * np.cos(x[1]) * \
        np.exp(-((x[0] - np.pi) ** 2) - (x[1] - np.pi) ** 2)
    return y


def eggholder(x: np.ndarray):
    """Egg Holder function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Egg Holder function at x.
    """
    y = -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[1] + 0.5 * x[0] + 47))) - \
        x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))
    return y


def griewank(x: np.ndarray):
    """Griewank function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Griewank function at x.
    """
    d = x.shape[0]
    i = np.arange(1, d + 1)
    y = 1 + np.sum(x**2 / 4000) - np.prod(np.cos(x / np.sqrt(i)))
    return y


def holder_table(x: np.ndarray):
    """Holder-Table function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Holder-Table function at x.
    """
    y = -np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(
        np.abs(1 - np.sqrt(x[0]**2 + x[1]**2) / np.pi)))
    return y


def levy(x: np.ndarray):
    """Levy function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Levy function at x.
    """
    z = 1 + (x - 1) / 4
    y = (np.sin(np.pi * z[0]) ** 2 + sum(
        (z[:-1] - 1) ** 2 * (1 + 10 * np.sin(
            np.pi * z[:-1] + 1) ** 2)) + (
                z[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * z[-1]) ** 2))
    return y


def rastrigin(x: np.ndarray):
    """Rastrigin function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Rastrigin function at x.
    """
    d = x.shape[0]
    y = 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    return y


def rosenbrock(x: np.ndarray):
    """Rosenbrock function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Rosenbrock function at x.
    """
    y = np.sum(
        np.abs(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))
    return y


def rotatedhyperellipsoid(x: np.ndarray):
    """Rotated Hyper-Ellipsoid function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Rotated Hyper-Ellipsoid function at x.
    """
    d = x.shape[0]
    y = np.sum([np.sum(x[: i + 1] ** 2) for i in range(d)])
    return y


def schwefel(x: np.ndarray):
    """Schwefel function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Schwefel function at x.
    """
    d = x.shape[0]
    y = 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    return y


def sphere(x: np.ndarray):
    """Sphere function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Sphere function at x.
    """
    y = np.sum(x**2)
    return y


def styblinskitang(x: np.ndarray):
    """Styblinski-Tang function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Styblinski-Tang function at x.
    """
    y = 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)
    return y


def threehump(x: np.ndarray):
    """Three-Hump function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Three-Hump function at x.
    """
    x1, x2 = x
    y = 2 * x1**2 - 1.05 * x1**4 + x1**6 * (1 / 6) + x1 * x2 + x2**2
    return y


def zakharov(x: np.ndarray):
    """Zakharov function

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Zakharov function at x.
    """
    d = x.shape[0]
    y = np.sum(x**2) + np.sum(0.5 * np.arange(1, d + 1) * x) ** 2 + \
        np.sum(0.5 * np.arange(1, d + 1) * x) ** 4
    return y

# =============================================================================


BENCHMARK_FUNCTIONS = {
    'ackley': ackley,
    'beale': beale,
    'bohachevsky': bohachevsky,
    'booth': booth,
    'branin': branin,
    'bukin': bukin,
    'crossintray': crossintray,
    'dixonprice': dixonprice,
    'dropwave': drop_wave,
    'easom': easom,
    'eggholder': eggholder,
    'griewank': griewank,
    'holdertable': holder_table,
    'levy': levy,
    'rastrigin': rastrigin,
    'rosenbrock': rosenbrock,
    'rotatedhyperellipsoid': rotatedhyperellipsoid,
    'schwefel': schwefel,
    'sphere': sphere,
    'styblinskitang': styblinskitang,
    'threehump': threehump,
    'zakharov': zakharov
}
