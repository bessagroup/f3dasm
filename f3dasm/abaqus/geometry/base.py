'''
Created on 2020-11-17 10:16:09
Last modified on 2020-11-24 19:23:11

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# abaqus
from abaqusConstants import FROM_SECTION

# standard library
from abc import ABCMeta
from abc import abstractmethod

# local library
from ..modelling.mesh import MeshGenerator


# object definition

class Geometry(object):
    __metaclass__ = ABCMeta

    def __init__(self, default_mesh=False):
        if default_mesh:
            self.mesh = MeshGenerator()

    def create_part(self, model):
        return None

    def create_instance(self, model):
        return None

    def create(self, model):
        self.create_part(model)
        self.create_instance(model)
        # TODO: add generate mesh

    def generate_mesh(self):
        return self.mesh.generate_mesh(self.part)

    def _assign_section(self, region, part=None):

        # initialization
        if part is None:
            part = self.part

        # assign section
        part.SectionAssignment(region=region,
                               sectionName=self.material.section.name,
                               thicknessAssignment=FROM_SECTION)


# TODO: default mesh strategy?
