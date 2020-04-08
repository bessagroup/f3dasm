'''
Created on 2020-04-08 12:03:11
Last modified on 2020-04-08 14:10:24
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

@collaborators:
    Rodrigo Tavares (em10140@fe.up.pt)

Main goal
---------
Define classes to create materials in Abaqus using information from the
material database.
'''


#%% imports

# abaqus
from abaqusConstants import (ENGINEERING_CONSTANTS, LAMINA)

# standard library
import abc

# local library
from ...material.material import Material


#%% abstract classes

class AbaqusMaterial(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, material, name, model, create_section):
        '''
        Parameters
        ----------
        material : instance of the class Material or str
            If str, it reads material from database.
        name : str
            If None, it uses material name.
        model : abaqus mdb object
        create_section : bool
            If an homogeneous section is created.
        '''
        if isinstance(material, str):
            material = Material(material, read=True)
        self.material = material
        self.name = material.name.upper() if name is None else name.upper()
        self.create_section = create_section
        # initialize variables
        self.abaqusMaterial = None
        # computations
        if model:
            self.create_material(model)

    def _add_density(self):
        # TODO: expand to deal with tabular data

        # verify if density info is available
        if not self.material.has_prop('density'):
            return

        # add density
        rho = self.material.get_value('density')
        self.abaqusMaterial.Density(table=((rho, ),))

    def _verify_existing_material(self, model):
        '''
        Notes
        -----
        -if name already exists in model materials, then it simply uses the
        defined material.
        '''

        # verify if method was already called
        if self.abaqusMaterial is not None:
            return True

        # verify existing material
        if self.name in model.materials.keys():
            self.abaqusMaterial = model.materials[self.name]
            return True

        return False

    @abc.abstractmethod
    def create_material(self, model):
        pass


class ElasticMaterial(AbaqusMaterial):
    __metaclass__ = abc.ABCMeta

    def __init__(self, material, name, model, create_section):
        AbaqusMaterial.__init__(self, material, name, model, create_section)

    def create_material(self, model):

        # TODO: expand to deal with tabular data

        # verify existing material
        if self._verify_existing_material(model):
            return

        # read required properties
        mechanical_constants = [self.material.get_value(name)
                                for name in self.required_mechanical_constants]

        # create material
        self.abaqusMaterial = model.Material(name=self.name)

        # define properties
        self.abaqusMaterial.Elastic(
            type=self.elastic_type, table=(mechanical_constants,))
        self._add_density()

        # define section
        if self.create_section:
            model.HomogeneousSolidSection(name=self.name, material=self.name,
                                          thickness=None)


#%% material definiton

class IsotropicMaterial(ElasticMaterial):

    req_material_orientation = False
    elastic_type = ENGINEERING_CONSTANTS
    required_mechanical_constants = ['E', 'nu']

    def __init__(self, material, name=None, model=None, create_section=True):
        ElasticMaterial.__init__(self, material, name, model, create_section)


class OrthotropicMaterial(ElasticMaterial):

    req_material_orientation = True
    elastic_type = ENGINEERING_CONSTANTS
    required_mechanical_constants = ['E1', 'E2', 'E3',
                                     'nu12', 'nu13', 'nu23',
                                     'G12', 'G13', 'G23']

    def __init__(self, material, name=None, model=None, create_section=True):
        ElasticMaterial.__init__(self, material, name, model, create_section)


class LaminaMaterial(ElasticMaterial):
    req_material_orientation = True
    elastic_type = LAMINA
    required_mechanical_constants = ['E1', 'E2', 'nu12',
                                     'G12', 'G13', 'G23']

    def __init__(self, material, name=None, model=None, create_section=False):
        ElasticMaterial.__init__(self, material, name, model, create_section)
