import numpy as np


def distancePointPoint(P1,P2):
    """Calculate the distance between to points P1 and P2"""
    distVec=np.atleast_2d(P2)-np.atleast_2d(P1)
    dist=np.linalg.norm(distVec,axis=1)

    return dist, distVec


def distancePointLine(P,A,B,finiteLine=True):
    """Calculate the distance between a point P and a line defined by points
    A and B. If A and B are the starting and ending points of the line, i.e.,
    the line is finite, set finiteLine flag to True."""

    distAB, AB = distancePointPoint(A,B)
    PA = np.atleast_2d(P)-np.atleast_2d(A)
    PB = np.atleast_2d(P)-np.atleast_2d(B)

    distVec=np.cross(PA,PB)/(distAB)
    if finiteLine:
        clostestPointIsA=np.where(np.dot(AB,-PA.T) < 0)
        clostestPointIsB=np.where(np.dot(AB,PB.T) < 0)
        distVec[clostestPointIsA]=PA[clostestPointIsA]
        distVec[clostestPointIsB]=PA[clostestPointIsB]

    return np.linalg.norm(distVec,axis=1), distVec


def distanceLineLine(A1,B1,A2,B2,finiteLine=True):
    """Calculate the distance between two lines each defined by points A_i and
    B_i. If A_i and B_i are the starting and end points of the lines, i.e., the
    lines are finite, set finiteLine flag to True"""
    pass
