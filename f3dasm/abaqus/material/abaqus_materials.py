'''
Created on 2020-04-08 12:03:11
Last modified on 2020-11-04 13:17:24

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
from abaqusConstants import (ISOTROPIC, ENGINEERING_CONSTANTS, LAMINA, ORTHOTROPIC)

# standard library
import abc


# TODO: add user material


# abaqus material class

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
        for av_material_behavior in GENERAL_MATERIAL_BEHAVIORS:
            for prop in av_material_behavior.required_props:
                if prop not in self.props.keys():
                    break
            else:
                material_behaviors.append(av_material_behavior())

        # add exclusive behaviors
        for MATERIAL_BEHAVIORS in [ELASTIC_MATERIAL_BEHAVIORS, EXPANSION_MATERIAL_BEHAVIORS]:
            for av_material_behavior in MATERIAL_BEHAVIORS:
                for prop in av_material_behavior.required_props:
                    if prop not in self.props.keys():
                        break
                else:
                    material_behaviors.append(av_material_behavior())
                    break

        return material_behaviors

    def getAbaqusMaterial(self, model):
        return model.materials[self.name]


# material behavior abstract class

class MaterialBehavior(object):
    '''
    Notes
    -----
    If the material behavior starts without `props`, then AbaqusMaterials must
    contain the required properties for the behavior.
    '''

    def __init__(self, props=None, **kwargs):
        self.props = None
        self.opt_props = None
        self.temperature = None
        self.kwargs = kwargs
        if props is not None:
            self.get_props_from_dict(props)

    def get_props_from_dict(self, props):

        # required props
        self.props = [props[name] for name in self.required_props]

        # temperature
        if 'T' in props.keys():
            self.temperature = props['T']

        # optional props
        if hasattr(self, 'optional_props') and props is not None:
            self.opt_props = {}
            for name, value in props.items():
                if name in self.optional_props:
                    self.opt_props[self.optional_props_map[name]] = value
            if len(self.opt_props) == 0:
                self.opt_props = None

    def _get_table(self):
        if not self.temperature or type(self.props[0]) not in [list, tuple]:
            return False, (self.props,)
        else:
            table = []
            for i in range(len(self.temperature)):
                table.append([prop[i] for prop in self.props] + [self.temperature[i]])
            return True, table

    def create_behavior(self, abaqusMaterial):

        # deal with required properties
        create_behavior = getattr(abaqusMaterial, self.method_name)
        kwargs = {}
        if hasattr(self, 'behavior_type'):
            kwargs['type'] = self.behavior_type
        temp_dep, table = self._get_table()
        if temp_dep:
            kwargs['temperatureDependency'] = True
        create_behavior(table=table, **kwargs)

        # deal with additional properties
        if self.opt_props is not None:
            self.setValues(abaqusMaterial, **self.opt_props)

        # deal with kwargs
        if self.kwargs:
            self.setValues(abaqusMaterial, **self.kwargs)

    def setValues(self, abaqusMaterial, **kwargs):
        set_values = getattr(abaqusMaterial, self.method_name.lower())
        set_values.setValues(**kwargs)

    @property
    def has_props(self):
        return not (self.props is None)


# density

class Density(MaterialBehavior):
    required_props = ['rho']
    method_name = 'Density'

    def __init__(self, props=None, **kwargs):
        super(Density, self).__init__(props, **kwargs)


# elastic behavior

class ElasticBehavior(MaterialBehavior):
    __metaclass__ = abc.ABCMeta
    method_name = 'Elastic'

    def __init__(self, props=None, **kwargs):
        super(ElasticBehavior, self).__init__(props, **kwargs)


class ElasticIsotropicBehavior(ElasticBehavior):
    req_material_orientation = False
    behavior_type = ISOTROPIC
    required_props = ['E', 'nu']

    def __init__(self, props=None, **kwargs):
        super(ElasticIsotropicBehavior, self).__init__(props, **kwargs)


class ElasticLaminaBehavior(ElasticBehavior):
    req_material_orientation = True
    behavior_type = LAMINA
    required_props = ['E1', 'E2', 'nu12',
                      'G12', 'G13', 'G23']

    def __init__(self, props=None, **kwargs):
        super(ElasticLaminaBehavior, self).__init__(props, **kwargs)


class ElasticEngineeringConstantsBehavior(ElasticBehavior):
    req_material_orientation = True
    behavior_type = ENGINEERING_CONSTANTS
    required_props = ['E1', 'E2', 'E3',
                      'nu12', 'nu13', 'nu23',
                      'G12', 'G13', 'G23']

    def __init__(self, props=None, **kwargs):
        super(ElasticEngineeringConstantsBehavior, self).__init__(props, **kwargs)


# expansion behavior

class ExpansionBehavior(MaterialBehavior):
    __metaclass__ = abc.ABCMeta
    method_name = 'Expansion'
    optional_props = ['T0']
    optional_props_map = {'T0': 'zero'}

    def __init__(self, props=None, **kwargs):
        super(ExpansionBehavior, self).__init__(props, **kwargs)


class ExpansionIsotropicBehavior(ExpansionBehavior):
    required_props = ['alpha']
    behavior_type = ISOTROPIC

    def __init__(self, props=None, **kwargs):
        super(ExpansionIsotropicBehavior, self).__init__(props, **kwargs)


class ExpansionOrthotropicBehavior(ExpansionBehavior):
    required_props = ['alpha1', 'alpha2', 'alpha3']
    behavior_type = ORTHOTROPIC

    def __init__(self, props=None, **kwargs):
        super(ExpansionOrthotropicBehavior, self).__init__(props, **kwargs)


GENERAL_MATERIAL_BEHAVIORS = [Density, ]

ELASTIC_MATERIAL_BEHAVIORS = [ElasticEngineeringConstantsBehavior,
                              ElasticLaminaBehavior, ElasticIsotropicBehavior, ]  # order by complexity

EXPANSION_MATERIAL_BEHAVIORS = [ExpansionOrthotropicBehavior, ExpansionIsotropicBehavior, ]
