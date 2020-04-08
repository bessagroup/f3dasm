'''
Created on 2020-04-08 10:34:36
Last modified on 2020-04-08 14:09:39
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

@collaborators:
    F. Otero (fotero@inegi.up.pt)

Main goal
---------
Define a general class for material and other classes.
'''


#%% imports

# local library
from .common import Property
from . import material_data  # it is used within eval


#%% define classes

class Material(object):

    def __init__(self, name, info=None, read=True):
        self.name = name  # name is updated when the file is read
        self.info = info
        # variable initialization
        self.filename = None
        self.props = {}
        # computations
        if read:
            self.read_material()

    def add_prop(self, name, value, deviation=None, unit=None, info=None):
        self.props[name] = Property(value, deviation, unit, info)

    def change_prop(self, name, value=None, deviation=None, unit=None, info=None):
        self._verify_if_has_prop(name, 'data')

        prop = self.props[name]
        prop.value = value if value is not None else prop.value
        prop.deviation = deviation if deviation is not None else prop.deviation
        prop.unit = unit if unit is not None else prop.unit
        prop.info = info if info is not None else prop.info

    def has_prop(self, name):
        return name in self.props.keys()

    def _verify_if_has_prop(self, prop_name, var):
        if prop_name not in self.props.keys():
            raise ValueError('Could not find {} in {} properties'.format(prop_name, self.name))
        elif getattr(self.props[prop_name], var) is None:
            raise ValueError('Could not find the {} of property {} in {} properties'.format(var, prop_name, self.name))
        return

    def get_data(self, prop_name):
        self._verify_if_has_prop(prop_name, 'data')

        return self.props[prop_name].data

    def get_value(self, prop_name):
        self._verify_if_has_prop(prop_name, 'value')

        return self.props[prop_name].value

    def get_deviation(self, prop_name):
        self._verify_if_has_prop(prop_name, 'deviation')

        return self.props[prop_name].deviation

    def get_unit(self, prop_name):
        self._verify_if_has_prop(prop_name, 'unit')

        return self.props[prop_name].unit

    def get_info(self, prop_name):
        self._verify_if_has_prop(prop_name, 'info')

        return self.props[prop_name].info

    def read_material(self):

        # initialization
        material = eval('material_data.%s' % self.name)

        # add properties
        for prop_name, value in material.items():
            if type(value) is Property:
                self.props[prop_name] = value
            else:  # assumes only value is given
                self.props[prop_name] = Property(value)
