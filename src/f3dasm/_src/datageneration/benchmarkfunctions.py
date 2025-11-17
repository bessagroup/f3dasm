#                                                                       Modules
# =============================================================================

# Third-party
import numpy as np

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


def ackley(
    x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi
):
    """Ackley function

    ![Ackley function surface](../img/functions/ackley.png){ width=30% }

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

    ![Beale function surface](../img/functions/beale.png){ width=30% }

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Beale function at x.
    """
    y = (
        (1.5 - x[0] + x[0] * x[1]) ** 2
        + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
    )
    return y


def bohachevsky(x: np.ndarray):
    """Bohachevsky N. 1 function

    ![Bohachevsky function surface](
        ../img/functions/bohachevsky.png){ width=30% }

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Bohachevsky N. 1 function at x.
    """
    y = (
        x[0] ** 2
        + 2 * x[1] ** 2
        - 0.3 * np.cos(3 * np.pi * x[0])
        - 0.4 * np.cos(4 * np.pi * x[1])
        + 0.7
    )
    return y


def booth(x: np.ndarray):
    """Booth function

    ![Booth function surface](../img/functions/booth.png){ width=30% }


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


def branin(
    x: np.ndarray,
    a=1,
    b=5.1 / (4 * np.pi**2),
    c=5 / np.pi,
    r=6,
    s=10,
    t=1 / (8 * np.pi),
):
    """Branin function

    ![Branin function surface](../img/functions/branin.png){ width=30% }


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
    y = a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2
    y = y + s * (1 - t) * np.cos(x[0]) + s
    return y


def bukin(x: np.ndarray):
    """Bukin N. 6 function

    ![Bukin function surface](../img/functions/bukin.png){ width=30% }


    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Bukin N. 6 function at x.
    """
    y = 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * np.abs(
        x[0] + 10
    )
    return y


def crossintray(x: np.ndarray):
    """Cross-in-Tray function

    ![Cross-in-tray function surface](
        ../img/functions/crossintray.png){ width=30% }


    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Cross-in-Tray function at x.
    """
    y = (
        -0.0001
        * (
            np.abs(np.sin(x[0]) * np.sin(x[1]))
            * np.exp(np.abs(100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))
            + 1
        )
        ** 0.1
    )
    return y


def dixonprice(x: np.ndarray):
    """Dixon Price function

    ![dixonprice function surface](
        ../img/functions/dixonprice.png){ width=30% }


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
        [(i + 1) * (2 * x[i] ** 2 - x[i - 1]) ** 2 for i in range(1, d)]
    )
    return y


def drop_wave(x: np.ndarray):
    """Drop-Wave function

    ![drop-wave function surface](../img/functions/drop_wave.png){ width=30% }


    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Drop-Wave function at x.
    """
    y = -(1 + np.cos(12 * np.sqrt(x[0] ** 2 + x[1] ** 2))) / (
        0.5 * (x[0] ** 2 + x[1] ** 2) + 2
    )
    return y


def easom(x: np.ndarray):
    """Easom function

    ![Easom function surface](../img/functions/easom.png){ width=30% }


    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Easom function at x.
    """
    y = (
        -np.cos(x[0])
        * np.cos(x[1])
        * np.exp(-((x[0] - np.pi) ** 2) - (x[1] - np.pi) ** 2)
    )
    return y


def eggholder(x: np.ndarray):
    """Egg Holder function

    ![Eggholder function surface](../img/functions/eggholder.png){ width=30% }


    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Egg Holder function at x.
    """
    y = -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[1] + 0.5 * x[0] + 47))) - x[
        0
    ] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))
    return y


def griewank(x: np.ndarray):
    """Griewank function

    ![griewank function surface](../img/functions/griewank.png){ width=30% }


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

    ![Holder-Table function surface](
        ../img/functions/holder_table.png){ width=30% }


    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Holder-Table function at x.
    """
    y = -np.abs(
        np.sin(x[0])
        * np.cos(x[1])
        * np.exp(np.abs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))
    )
    return y


def levy(x: np.ndarray):
    """Levy function

    ![Levy function surface](../img/functions/levy.png){ width=30% }


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
    y = (
        np.sin(np.pi * z[0]) ** 2
        + sum((z[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * z[:-1] + 1) ** 2))
        + (z[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * z[-1]) ** 2)
    )
    return y


def rastrigin(x: np.ndarray):
    """Rastrigin function

    ![Rastrigin function surface](../img/functions/rastrigin.png){ width=30% }


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

    ![Rosenbrock function surface](
        ../img/functions/rosenbrock.png){ width=30% }


    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The value of the Rosenbrock function at x.
    """
    y = np.sum(np.abs(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))
    return y


def rotatedhyperellipsoid(x: np.ndarray):
    """Rotated Hyper-Ellipsoid function

    ![Rotated Hyper-Ellipsoid function surface](
        ../img/functions/rotatedhyperellipsoid.png){ width=30% }


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

    ![Schwefel function surface](../img/functions/schwefel.png){ width=30% }


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
    """Sphere funct

    ![Sphere function surface](
        ../img/functions/sphere.png){ width=30% }


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

    ![styblinski-tang function surface](
        ../img/functions/styblinskitang.png){ width=30% }


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

    ![Threehump function surface](../img/functions/threehump.png){ width=30% }


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

    ![Zakharov function surface](../img/functions/zakharov.png){ width=30% }


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
    y = (
        np.sum(x**2)
        + np.sum(0.5 * np.arange(1, d + 1) * x) ** 2
        + np.sum(0.5 * np.arange(1, d + 1) * x) ** 4
    )
    return y


# =============================================================================


BENCHMARK_FUNCTIONS = {
    "ackley": ackley,
    "beale": beale,
    "bohachevsky": bohachevsky,
    "booth": booth,
    "branin": branin,
    "bukin": bukin,
    "crossintray": crossintray,
    "dixonprice": dixonprice,
    "dropwave": drop_wave,
    "easom": easom,
    "eggholder": eggholder,
    "griewank": griewank,
    "holdertable": holder_table,
    "levy": levy,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "rotatedhyperellipsoid": rotatedhyperellipsoid,
    "schwefel": schwefel,
    "sphere": sphere,
    "styblinskitang": styblinskitang,
    "threehump": threehump,
    "zakharov": zakharov,
}

BENCHMARK_BOUNDS = {
    "ackley": (-32.768, 32.768),
    "beale": (-4.5, 4.5),
    "bohachevsky": (-100, 100),
    "booth": (-10, 10),
    "branin": (-5, 10),  # is actually  [-5, 10] x [0, 15]
    "bukin": (-15, -5),
    "crossintray": (-10, 10),
    "dixonprice": (-10, 10),
    "dropwave": (-5.12, 5.12),
    "easom": (-100, 100),
    "eggholder": (-512, 512),
    "griewank": (-600, 600),
    "holdertable": (-10, 10),
    "levy": (-10, 10),
    "rastrigin": (-5.12, 5.12),
    "rosenbrock": (-5, 10),
    "rotatedhyperellipsoid": (-65.536, 65.536),
    "schwefel": (-500, 500),
    "sphere": (-5.12, 5.12),
    "styblinskitang": (-5, 5),
    "threehump": (-5, 5),
    "zakharov": (-5, 10),
}
