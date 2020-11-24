'''
Created on 2020-11-17 10:16:09
Last modified on 2020-11-24 11:46:23

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# standard library
from abc import ABCMeta
from abc import abstractmethod

# local library
from ..modelling.mesh import MeshGenerator


# object definition

class Geometry(object):
    __metaclass__ = ABCMeta

    def __init_(self, default_mesh=False):
        if default_mesh:
            self.mesh = MeshGenerator()

    @abstractmethod
    def create_part(self, model):
        return None

    def create_instance(self, model):
        return None

    def create(self, model):
        self.create_part(model)
        self.create_instance(model)
        # TODO: add generate mesh


# TODO: default mesh strategy?
