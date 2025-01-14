"""
Functional aliasses for the builtin functions.
"""
#                                                                       Modules
# =============================================================================

# Standard
from functools import partial
from typing import Optional

# Third-party
import numpy as np

# Local
from ...core import DataGenerator
from .pybenchfunction import (Ackley, AckleyN2, AckleyN3, AckleyN4, Adjiman,
                              Bartels, Beale, Bird, BohachevskyN1,
                              BohachevskyN2, BohachevskyN3, Booth, Branin,
                              Brent, Brown, BukinN6, Colville, CrossInTray,
                              DeckkersAarts, DeJongN5, DixonPrice, DropWave,
                              Easom, EggCrate, EggHolder, Exponential,
                              GoldsteinPrice, Griewank, HappyCat, Himmelblau,
                              HolderTable, Keane, Langermann, Leon, Levy,
                              LevyN13, Matyas, McCormick, Michalewicz,
                              Periodic, Powell, Qing, Quartic, Rastrigin,
                              Ridge, Rosenbrock, RotatedHyperEllipsoid,
                              Salomon, SchaffelN1, SchaffelN2, SchaffelN3,
                              SchaffelN4, Schwefel, Schwefel2_20, Schwefel2_21,
                              Schwefel2_22, Schwefel2_23, Shekel, Shubert,
                              ShubertN3, ShubertN4, Sphere, StyblinskiTang,
                              SumSquares, Thevenot, ThreeHump, Trid, Wolfe,
                              XinSheYang, XinSheYangN2, XinSheYangN3,
                              XinSheYangN4, Zakharov)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def fn(base_class, scale_bounds: Optional[np.ndarray] = None,
       noise: Optional[float] = None, offset: bool = True,
       seed: Optional[int] = None) -> DataGenerator:
    """
    Parameters
    ----------
    scale_bounds : Optional[np.ndarray], optional
        Bounds for scaling the input data, by default None.
    noise : Optional[float], optional
        Amount of noise to add to the generated data, by default None.
    offset : bool, optional
        Whether to apply an offset to the input data, by default True.
    seed : Optional[int], optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    DataGenerator
        A DataGenerator configured with the provided
        options.
    """
    return base_class(scale_bounds=scale_bounds, noise=noise, offset=offset,
                      seed=seed)
# =============================================================================


ackley = partial(fn, base_class=Ackley)
ackley.__doc__ = f"{Ackley.__doc__}\n\n{fn.__doc__}"

ackleyn2 = partial(fn, base_class=AckleyN2)
ackleyn2.__doc__ = f"{AckleyN2.__doc__}\n\n{fn.__doc__}"

ackleyn3 = partial(fn, base_class=AckleyN3)
ackleyn3.__doc__ = f"{AckleyN3.__doc__}\n\n{fn.__doc__}"

ackleyn4 = partial(fn, base_class=AckleyN4)
ackleyn4.__doc__ = f"{AckleyN4.__doc__}\n\n{fn.__doc__}"

adjiman = partial(fn, base_class=Adjiman)
adjiman.__doc__ = f"{Adjiman.__doc__}\n\n{fn.__doc__}"

bartels = partial(fn, base_class=Bartels)
bartels.__doc__ = f"{Bartels.__doc__}\n\n{fn.__doc__}"

beale = partial(fn, base_class=Beale)
beale.__doc__ = f"{Beale.__doc__}\n\n{fn.__doc__}"

bird = partial(fn, base_class=Bird)
bird.__doc__ = f"{Bird.__doc__}\n\n{fn.__doc__}"

# Manually creating the rest of the partial functions
bohachevskyn1 = partial(fn, base_class=BohachevskyN1)
bohachevskyn1.__doc__ = f"{BohachevskyN1.__doc__}\n\n{fn.__doc__}"

bohachevskyn2 = partial(fn, base_class=BohachevskyN2)
bohachevskyn2.__doc__ = f"{BohachevskyN2.__doc__}\n\n{fn.__doc__}"

bohachevskyn3 = partial(fn, base_class=BohachevskyN3)
bohachevskyn3.__doc__ = f"{BohachevskyN3.__doc__}\n\n{fn.__doc__}"

booth = partial(fn, base_class=Booth)
booth.__doc__ = f"{Booth.__doc__}\n\n{fn.__doc__}"

branin = partial(fn, base_class=Branin)
branin.__doc__ = f"{Branin.__doc__}\n\n{fn.__doc__}"

brent = partial(fn, base_class=Brent)
brent.__doc__ = f"{Brent.__doc__}\n\n{fn.__doc__}"

brown = partial(fn, base_class=Brown)
brown.__doc__ = f"{Brown.__doc__}\n\n{fn.__doc__}"

bukinn6 = partial(fn, base_class=BukinN6)
bukinn6.__doc__ = f"{BukinN6.__doc__}\n\n{fn.__doc__}"

colville = partial(fn, base_class=Colville)
colville.__doc__ = f"{Colville.__doc__}\n\n{fn.__doc__}"

crossintray = partial(fn, base_class=CrossInTray)
crossintray.__doc__ = f"{CrossInTray.__doc__}\n\n{fn.__doc__}"

deckkersaarts = partial(fn, base_class=DeckkersAarts)
deckkersaarts.__doc__ = f"{DeckkersAarts.__doc__}\n\n{fn.__doc__}"

dejongn5 = partial(fn, base_class=DeJongN5)
dejongn5.__doc__ = f"{DeJongN5.__doc__}\n\n{fn.__doc__}"

dixonprice = partial(fn, base_class=DixonPrice)
dixonprice.__doc__ = f"{DixonPrice.__doc__}\n\n{fn.__doc__}"

dropwave = partial(fn, base_class=DropWave)
dropwave.__doc__ = f"{DropWave.__doc__}\n\n{fn.__doc__}"

easom = partial(fn, base_class=Easom)
easom.__doc__ = f"{Easom.__doc__}\n\n{fn.__doc__}"

eggcrate = partial(fn, base_class=EggCrate)
eggcrate.__doc__ = f"{EggCrate.__doc__}\n\n{fn.__doc__}"

eggholder = partial(fn, base_class=EggHolder)
eggholder.__doc__ = f"{EggHolder.__doc__}\n\n{fn.__doc__}"

exponential = partial(fn, base_class=Exponential)
exponential.__doc__ = f"{Exponential.__doc__}\n\n{fn.__doc__}"

goldsteinprice = partial(fn, base_class=GoldsteinPrice)
goldsteinprice.__doc__ = f"{GoldsteinPrice.__doc__}\n\n{fn.__doc__}"

griewank = partial(fn, base_class=Griewank)
griewank.__doc__ = f"{Griewank.__doc__}\n\n{fn.__doc__}"

happycat = partial(fn, base_class=HappyCat)
happycat.__doc__ = f"{HappyCat.__doc__}\n\n{fn.__doc__}"

himmelblau = partial(fn, base_class=Himmelblau)
himmelblau.__doc__ = f"{Himmelblau.__doc__}\n\n{fn.__doc__}"

holdertable = partial(fn, base_class=HolderTable)
holdertable.__doc__ = f"{HolderTable.__doc__}\n\n{fn.__doc__}"

keane = partial(fn, base_class=Keane)
keane.__doc__ = f"{Keane.__doc__}\n\n{fn.__doc__}"

langermann = partial(fn, base_class=Langermann)
langermann.__doc__ = f"{Langermann.__doc__}\n\n{fn.__doc__}"

leon = partial(fn, base_class=Leon)
leon.__doc__ = f"{Leon.__doc__}\n\n{fn.__doc__}"

levy = partial(fn, base_class=Levy)
levy.__doc__ = f"{Levy.__doc__}\n\n{fn.__doc__}"

levyn13 = partial(fn, base_class=LevyN13)
levyn13.__doc__ = f"{LevyN13.__doc__}\n\n{fn.__doc__}"

matyas = partial(fn, base_class=Matyas)
matyas.__doc__ = f"{Matyas.__doc__}\n\n{fn.__doc__}"

mccormick = partial(fn, base_class=McCormick)
mccormick.__doc__ = f"{McCormick.__doc__}\n\n{fn.__doc__}"

michalewicz = partial(fn, base_class=Michalewicz)
michalewicz.__doc__ = f"{Michalewicz.__doc__}\n\n{fn.__doc__}"

periodic = partial(fn, base_class=Periodic)
periodic.__doc__ = f"{Periodic.__doc__}\n\n{fn.__doc__}"

powell = partial(fn, base_class=Powell)
powell.__doc__ = f"{Powell.__doc__}\n\n{fn.__doc__}"

qing = partial(fn, base_class=Qing)
qing.__doc__ = f"{Qing.__doc__}\n\n{fn.__doc__}"

quartic = partial(fn, base_class=Quartic)
quartic.__doc__ = f"{Quartic.__doc__}\n\n{fn.__doc__}"

rastrigin = partial(fn, base_class=Rastrigin)
rastrigin.__doc__ = f"{Rastrigin.__doc__}\n\n{fn.__doc__}"

ridge = partial(fn, base_class=Ridge)
ridge.__doc__ = f"{Ridge.__doc__}\n\n{fn.__doc__}"

rosenbrock = partial(fn, base_class=Rosenbrock)
rosenbrock.__doc__ = f"{Rosenbrock.__doc__}\n\n{fn.__doc__}"

rotatedhyperellipsoid = partial(fn, base_class=RotatedHyperEllipsoid)
rotatedhyperellipsoid.__doc__ = (
    f"{RotatedHyperEllipsoid.__doc__}\n\n{fn.__doc__}")

salomon = partial(fn, base_class=Salomon)
salomon.__doc__ = f"{Salomon.__doc__}\n\n{fn.__doc__}"

schaffeln1 = partial(fn, base_class=SchaffelN1)
schaffeln1.__doc__ = f"{SchaffelN1.__doc__}\n\n{fn.__doc__}"

schaffeln2 = partial(fn, base_class=SchaffelN2)
schaffeln2.__doc__ = f"{SchaffelN2.__doc__}\n\n{fn.__doc__}"

schaffeln3 = partial(fn, base_class=SchaffelN3)
schaffeln3.__doc__ = f"{SchaffelN3.__doc__}\n\n{fn.__doc__}"

schaffeln4 = partial(fn, base_class=SchaffelN4)
schaffeln4.__doc__ = f"{SchaffelN4.__doc__}\n\n{fn.__doc__}"

schwefel = partial(fn, base_class=Schwefel)
schwefel.__doc__ = f"{Schwefel.__doc__}\n\n{fn.__doc__}"

schwefel2_20 = partial(fn, base_class=Schwefel2_20)
schwefel2_20.__doc__ = f"{Schwefel2_20.__doc__}\n\n{fn.__doc__}"

schwefel2_21 = partial(fn, base_class=Schwefel2_21)
schwefel2_21.__doc__ = f"{Schwefel2_21.__doc__}\n\n{fn.__doc__}"

schwefel2_22 = partial(fn, base_class=Schwefel2_22)
schwefel2_22.__doc__ = f"{Schwefel2_22.__doc__}\n\n{fn.__doc__}"

schwefel2_23 = partial(fn, base_class=Schwefel2_23)
schwefel2_23.__doc__ = f"{Schwefel2_23.__doc__}\n\n{fn.__doc__}"

shekel = partial(fn, base_class=Shekel)
shekel.__doc__ = f"{Shekel.__doc__}\n\n{fn.__doc__}"

shubert = partial(fn, base_class=Shubert)
shubert.__doc__ = f"{Shubert.__doc__}\n\n{fn.__doc__}"

shubertn3 = partial(fn, base_class=ShubertN3)
shubertn3.__doc__ = f"{ShubertN3.__doc__}\n\n{fn.__doc__}"

shubertn4 = partial(fn, base_class=ShubertN4)
shubertn4.__doc__ = f"{ShubertN4.__doc__}\n\n{fn.__doc__}"

sphere = partial(fn, base_class=Sphere)
sphere.__doc__ = f"{Sphere.__doc__}\n\n{fn.__doc__}"

styblinskitang = partial(fn, base_class=StyblinskiTang)
styblinskitang.__doc__ = f"{StyblinskiTang.__doc__}\n\n{fn.__doc__}"

sumsquares = partial(fn, base_class=SumSquares)
sumsquares.__doc__ = f"{SumSquares.__doc__}\n\n{fn.__doc__}"

thevenot = partial(fn, base_class=Thevenot)
thevenot.__doc__ = f"{Thevenot.__doc__}\n\n{fn.__doc__}"

threehump = partial(fn, base_class=ThreeHump)
threehump.__doc__ = f"{ThreeHump.__doc__}\n\n{fn.__doc__}"

trid = partial(fn, base_class=Trid)
trid.__doc__ = f"{Trid.__doc__}\n\n{fn.__doc__}"

wolfe = partial(fn, base_class=Wolfe)
wolfe.__doc__ = f"{Wolfe.__doc__}\n\n{fn.__doc__}"

xin_she_yang = partial(fn, base_class=XinSheYang)
xin_she_yang.__doc__ = f"{XinSheYang.__doc__}\n\n{fn.__doc__}"

xin_she_yang2 = partial(fn, base_class=XinSheYangN2)
xin_she_yang2.__doc__ = f"{XinSheYangN2.__doc__}\n\n{fn.__doc__}"

xin_she_yang3 = partial(fn, base_class=XinSheYangN3)
xin_she_yang3.__doc__ = f"{XinSheYangN3.__doc__}\n\n{fn.__doc__}"

xin_she_yang4 = partial(fn, base_class=XinSheYangN4)
xin_she_yang4.__doc__ = f"{XinSheYangN4.__doc__}\n\n{fn.__doc__}"

zakharov = partial(fn, base_class=Zakharov)
zakharov.__doc__ = f"{Zakharov.__doc__}\n\n{fn.__doc__}"
