'''
Created on 2020-03-24 15:01:13
Last modified on 2020-03-24 16:28:53
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Define functions used among the modules of the current subpackage.
'''


#%% imports

# third-party
import numpy as np


#%% geometry-related functions

def transform_point(point, orientation=0, rotation_axis=3,
                    origin_translation=(0., 0.), origin=(0., 0.),):
    '''
    Transform point given the orientation.

    Parameters
    ----------
    point : array-like, shape = [2] or [3]
    orientation : float
        Angle between x_min and x_min' (radians).
    rotation_axis : int, possible values: 1, 2 , 3
        Axis about the rotation is performed.
    origin_translation : array-like, shape = [2]
        Translation of the origin.
    origin : array-like, shape = [2]
        Origin of initial axis.


    Returns
    -------
    transformed_point : array-like, shape = [2] or [3]
        Transformed point.
    '''

    # initialization
    m, n = np.cos(orientation), np.sin(orientation)
    cx, cy = origin
    d = len(point)
    axes = (1, 2, 3)
    if d == 2:
        i, j = 0, 1
    else:
        i, j = [ii - 1 for ii in axes if ii != rotation_axis]

    # computation
    transformed_coordinates = (origin_translation[0] + cx + (point[i] - cx) * m - (point[j] - cy) * n,
                               origin_translation[1] + cy + (point[i] - cx) * n + (point[j] - cy) * m)

    # transformed point
    transformed_point = [None] * d
    transformed_point[i] = transformed_coordinates[0]
    transformed_point[j] = transformed_coordinates[1]
    if d == 3:
        transformed_point[rotation_axis - 1] = point[rotation_axis - 1]

    return transformed_point


def get_orientations_360(orientation, addition_angle=np.pi / 2.):
    orientations = [orientation]
    stop_angle = 0.
    while stop_angle < 2 * np.pi - addition_angle:
        orientations.append(orientations[-1] + addition_angle)
        stop_angle += addition_angle
    return orientations
