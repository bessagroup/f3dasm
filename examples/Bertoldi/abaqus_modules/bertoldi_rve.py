'''
Created on 2020-10-15 08:38:20
Last modified on 2020-11-04 16:41:10

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# third-party
import numpy as np
from f3dasm.abaqus.geometry.shapes import Shape
from f3dasm.abaqus.geometry.utils import transform_point


# object definition

class BertoldiPore(Shape):

    def __init__(self, center, r_0, c_1, c_2, n_points=100):
        self.center = center
        self.r_0 = r_0
        self.c_1 = c_1
        self.c_2 = c_2
        self.n_points = n_points

    def create_inner_geometry(self, sketch, *args, **kwargs):
        '''
        Creates particular geometry of this example, i.e. internal pores.
        '''

        # initialization
        thetas = np.linspace(0., 2 * np.pi, self.n_points)

        # get points for spline
        points = []
        for theta in thetas:
            rr = self.r_0 * (1. + self.c_1 * np.cos(4. * theta) + self.c_2 * np.cos(8. * theta))
            points.append(transform_point((rr * np.cos(theta), rr * np.sin(theta)),
                                          origin_translation=self.center))

        # generate spline
        sketch.Spline(points=points)
