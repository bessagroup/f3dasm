'''
Created on 2020-04-08 12:03:11
Last modified on 2020-11-02 11:59:58

@author: L. F. Pereira (lfpereira@fe.up.pt)

@contributors:
    Rodrigo Tavares (em10140@fe.up.pt)

Main goal
---------
Define classes to create materials in Abaqus using information from the
material database.
'''


# imports

# abaqus
from abaqusConstants import (ISOTROPIC, ENGINEERING_CONSTANTS, LAMINA)

# standard library
import abc


# abaqus material classe

class AbaqusMaterial(object):

    def __init__(self, name, info=None, create_section=True, props=None,
                 material_behaviors=None, model=None):
        '''
        Parameters
        ----------
        name : str
            If None, it uses material name.
        props : dict
        material_behaviors: array-like
            If None, then the behaviors are automatically found out based on
            the available properties.
        model : abaqus mdb object
        create_section : bool
            If an homogeneous section is created.
        '''
        self.name = name
        self.info = info
        self.material_behaviors = material_behaviors
        self.create_section = create_section
        self.props = props
        # deal with behaviors
        if material_behaviors is None:
            self.material_behaviors = self._find_material_behaviors()
        else:
            self.material_behaviors = material_behaviors
        if model:
            self.create_material(model)

    def _verify_existing_material(self, model):
        '''
        Notes
        -----
        If name already exists in model materials, then it simply uses the
        defined material.
        '''

        # verify existing material
        if self.name in model.materials.keys():
            return True

        return False

    def create_material(self, model):

        # verify existing material
        if self._verify_existing_material(model):
            return

        # create material
        abaqusMaterial = model.Material(name=self.name)

        # create material behaviours
        for material_behavior in self.material_behaviors:
            if not material_behavior.has_props:
                material_behavior.get_props_from_dict(self.props)
            material_behavior.create_behavior(abaqusMaterial)

        # define section
        if self.create_section:
            model.HomogeneousSolidSection(name=self.name, material=self.name,
                                          thickness=None)

    def _find_material_behaviors(self):
        material_behaviors = []

        # add non-exclusive behaviors
        for av_material_behavior in OTHER_MATERIAL_BEHAVIORS:
            for prop in av_material_behavior.required_props:
                if prop not in self.props.keys():
                    break
            else:
                material_behaviors.append(av_material_behavior())

        # add exclusive behaviors
        for av_material_behavior in ELASTIC_MATERIAL_BEHAVIORS:
            for prop in av_material_behavior.required_props:
                if prop not in self.props.keys():
                    break
            else:
                material_behaviors.append(av_material_behavior())
                break

        return material_behaviors


# material behavior abstract class

class MaterialBehavior(object):
    '''
    Notes
    -----
    If the material behavior starts without `props`, then AbaqusMaterials must
    contain the required properties for the behavior.
    '''

    def __init__(self, props=None):
        self.props = None if props is None else [props[name] for name in self.required_props]

    def get_props_from_dict(self, props):
        self.props = [props[name] for name in self.required_props]

    @property
    def has_props(self):
        return not (self.props is None)


# density

class Density(MaterialBehavior):
    required_props = ['rho']

    def __init__(self, props=None):
        MaterialBehavior.__init__(self, props)

    def create_behavior(self, abaqusMaterial):
        abaqusMaterial.Density(table=((self.props[0], ),))


# elastic behavior

class ElasticBehavior(MaterialBehavior):
    __metaclass__ = abc.ABCMeta

    def __init__(self, props=None):
        MaterialBehavior.__init__(self, props)

    def create_behavior(self, abaqusMaterial):

        # define properties
        abaqusMaterial.Elastic(type=self.elastic_type, table=(self.props,))


class IsotropicBehavior(ElasticBehavior):
    req_material_orientation = False
    elastic_type = ISOTROPIC
    required_props = ['E', 'nu']

    def __init__(self, props=None):
        ElasticBehavior.__init__(self, props)


class LaminaBehavior(ElasticBehavior):
    req_material_orientation = True
    elastic_type = LAMINA
    required_props = ['E1', 'E2', 'nu12',
                      'G12', 'G13', 'G23']

    def __init__(self, props=None):
        ElasticBehavior.__init__(self, props)


class EngineeringConstantsBehavior(ElasticBehavior):
    req_material_orientation = True
    elastic_type = ENGINEERING_CONSTANTS
    required_props = ['E1', 'E2', 'E3',
                      'nu12', 'nu13', 'nu23',
                      'G12', 'G13', 'G23']

    def __init__(self, props=None):
        ElasticBehavior.__init__(self, props)


OTHER_MATERIAL_BEHAVIORS = [Density, ]

ELASTIC_MATERIAL_BEHAVIORS = [EngineeringConstantsBehavior, LaminaBehavior,
                              IsotropicBehavior, ]  # order by complexity
