'''
Created on 2020-04-08 10:34:36
Last modified on 2020-11-02 09:53:44

@author: L. F. Pereira (lfpereira@fe.up.pt)

@collaborators:
    F. Otero (fotero@inegi.up.pt)

Main goal
---------
Define a general class for material and other classes.
'''


# imports

# standard library
import functools

# local library
from . import material_data  # it is used within eval


# TODO: see composites
# TODO: add delete prop
# TODO: add filenames/dict_names and update read_material()
# TODO: maybe material can be really simplified (deviation, unit and info are required for the object? I think they should be only stored in the dictionary and that's it!)

# define classes

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

    def has_prop(self, name):
        return name in self.props.keys()

    @_Decorators._verify_if_has_prop('data')
    def change_prop(self, name, value=None, deviation=None, unit=None, info=None):
        prop = self.props[name]
        prop.value = value if value is not None else prop.value
        prop.deviation = deviation if deviation is not None else prop.deviation
        prop.unit = unit if unit is not None else prop.unit
        prop.info = info if info is not None else prop.info

    @_Decorators._verify_if_has_prop('data')
    def get_data(self, prop_name):
        return self.props[prop_name].data

    @_Decorators._verify_if_has_prop('value')
    def get_value(self, prop_name):
        return self.props[prop_name].value

    @_Decorators._verify_if_has_prop('deviation')
    def get_deviation(self, prop_name):
        return self.props[prop_name].deviation

    @_Decorators._verify_if_has_prop('unit')
    def get_unit(self, prop_name):
        return self.props[prop_name].unit

    @_Decorators._verify_if_has_prop('info')
    def get_info(self, prop_name):
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
