'''
Created on 2020-04-08 16:52:37
Last modified on 2020-04-20 21:46:22
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Define abaqus boundary conditions.

References
----------
1. Simulia (2015). ABAQUS 2016: Scripting Reference Guide
'''


#%% imports

# abaqus
from abaqusConstants import (UNSET, UNIFORM, NOT_APPLICABLE, OFF)

# standard library
import abc


#%% abstract classes

class BoundaryCondition(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, createStepName, region, model=None):
        '''
        Parameters
        ----------
        name : str
        createStepName : str
            Step in which the boundary condition is created.
        region : str or method or abaqus Set object
            Region to which the boundary condition is applied.
        model : abaqus mdb object
        '''
        self.name = name
        self.createStepName = createStepName
        self.region = region
        # computations
        if model:
            self.apply_bc(model)

    def _get_loaded_region(self, model):
        return model.rootAssembly.sets[self.region]

    def apply_bc(self, model):

        # get method
        apply_bc = getattr(model, self.method_name)

        # get loaded region
        if type(self.region) is str:
            loaded_region = self._get_loaded_region(model)
        elif callable(self.region):
            loaded_region = self.region(model)
        else:
            loaded_region = self.region

        # apply bc
        apply_bc(name=self.name, createStepName=self.createStepName,
                 region=loaded_region, **self.args)


#%% particular boundary conditions definition

class DisplacementBC(BoundaryCondition):

    method_name = 'DisplacementBC'

    def __init__(self, name, createStepName, region, model=None,
                 fieldName='', u1=UNSET, u2=UNSET, u3=UNSET, ur1=UNSET,
                 ur2=UNSET, ur3=UNSET, fixed=OFF, amplitude=UNSET,
                 distributionType=UNIFORM, localCsys=None, buckleCase=NOT_APPLICABLE):
        '''
        Parameters
        ----------
        fieldName : str
            Name of the AnalyticalField or DiscreteField object associated with
            this boundary condition. Applicable if distributionType=FIELD or
            distributionType=DISCRETE_FIELD.
        u1, u2, u3 : float, complex or abaqus constant
            Displacement component in i-direction (i=1, 2, 3).
        ur1, ur2, ur3 : float, complex or abaqus constant
            Rotation displacement component about the i-direction (i=1, 2, 3).
        fixed : bool
            Whether the boundary condition should remain fixed at the current
            values at the start of the step.
        amplitude : str or abaqus constant
            Name of the amplitude reference.
        distributionType : abaqus constant
            How the boundary condition is distributed spacially.
        localCsys : None or DatumCsys abaqus object
            Local coordinate system of the boundary condition's degrees of
            freedom.
        buckleCase : abaqus constant
            How the boundary condition is defined in a BUCKLE analysis.

        Notes
        -----
        -for further informations see p9-49 of [1].
        '''
        # create args dict
        self.args = {'fieldName': fieldName,
                     'u1': u1, 'u2': u2, 'u3': u3,
                     'ur1': ur1, 'ur2': ur2, 'ur3': ur3,
                     'fixed': fixed,
                     'amplitude': amplitude,
                     'distributionType': distributionType,
                     'localCsys': localCsys,
                     'buckleCase': buckleCase}
        # initialize parent
        BoundaryCondition.__init__(self, name, createStepName, region,
                                   model=model)


class ConcentratedForce(BoundaryCondition):

    method_name = 'ConcentratedForce'

    def __init__(self, name, createStepName, region, model=None,
                 cf1=UNSET, cf2=UNSET, cf3=UNSET, amplitude=UNSET,
                 follower=OFF, distributionType=UNIFORM, localCsys=None,
                 field=''):
        '''
        Parameters
        ----------
        cf1, cf2, cf3 : float or complex
            Load component in the i-direction (i=1, 2, 3).
        amplitude : str or abaqus constant
            Name of the amplitude reference.
        follower : bool
            Whether the direction of the force rotates with the rotation of the
            node.
        distributionType : abaqus constant
            How the boundary condition is distributed spacially.
        localCsys : None or DatumCsys abaqus object
            Local coordinate system of the boundary condition's degrees of
            freedom.
        field : str
            Name of the AnalyticalField or DiscreteField object associated with
            this boundary condition. Applicable if distributionType=FIELD.

        Notes
        -----
        -for further informations see p27-43 of [1].
        '''
        # create args dict
        self.args = {'cf1': cf1, 'cf2': cf2, 'cf3': cf3,
                     'amplitude': amplitude,
                     'follower': follower,
                     'distributionType': distributionType,
                     'localCsys': localCsys,
                     'field': field}
        # initialize parent
        BoundaryCondition.__init__(self, name, createStepName, region,
                                   model=model)


class Moment(BoundaryCondition):

    method_name = 'Moment'

    def __init__(self, name, createStepName, region, model=None,
                 cm1=UNSET, cm2=UNSET, cm3=UNSET, amplitude=UNSET,
                 follower=OFF, distributionType=UNIFORM, localCsys=None,
                 field=''):
        '''
        Parameters
        ----------
        cm1, cm2, cm3 : float or complex
            Load component in the i-direction (i=1, 2, 3).
        amplitude : str or abaqus constant
            Name of the amplitude reference.
        follower : bool
            Whether the direction of the force rotates with the rotation of the
            node.
        distributionType : abaqus constant
            How the boundary condition is distributed spacially.
        localCsys : None or DatumCsys abaqus object
            Local coordinate system of the boundary condition's degrees of
            freedom.
        field : str
            Name of the AnalyticalField or DiscreteField object associated with
            this boundary condition. Applicable if distributionType=FIELD.

        Notes
        -----
        -for further informations see p27-90 of [1].
        '''
        # create args dict
        self.args = {'cm1': cm1, 'cm2': cm2, 'cm3': cm3,
                     'amplitude': amplitude,
                     'follower': follower,
                     'distributionType': distributionType,
                     'localCsys': localCsys,
                     'field': field}
        # initialize parent
        BoundaryCondition.__init__(self, name, createStepName, region,
                                   model=model)
