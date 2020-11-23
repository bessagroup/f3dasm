'''
Created on 2020-11-23 09:14:24
Last modified on 2020-11-23 10:00:30

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# standard library
from abc import ABCMeta


# create objects

class Section(object):
    __metaclass__ = ABCMeta

    def __init__(self, name=None):
        self.name = None

    def create(self, model):

        # get section
        create_section = getattr(model, self.method_name)

        # create section
        create_section(**self.kwargs)


class HomogeneousSolidSection(Section):
    method_name = 'HomogeneousSolidSection'

    def __init__(self, name=None, material=None, thickness=None):
        super(HomogeneousSolidSection, self).__init__(name=name)
        self.material = material
        self.thickness = None

    @property
    def kwargs(self):
        kwargs = {'name': self.name, 'material': self.material,
                  'thickness': self.thickness}
        return kwargs
