'''
Created on 2020-11-17 10:16:09
Last modified on 2020-11-17 10:19:22

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# standard-library
from abc import ABCMeta
from abc import abstractmethod


# object definition

class Geometry(object):

    @abstractmethod
    def create_part(self, model):
        pass

    @abstractmethod
    def create_instance(self, model):
        pass

    def create(self, model):
        self.create_part(model)
        self.create_instance(model)
        # TODO: add generate mesh
