'''
Created on 2020-04-21 12:29:20
Last modified on 2020-04-21 14:56:19
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Define abaqus interaction properties.

References
----------
1. Simulia (2015). ABAQUS 2016: Scripting Reference Guide
'''


#%% imports

# abaqus
from abaqusConstants import (FRICTIONLESS, ISOTROPIC, OFF, COEFFICIENTS,
                             FRACTION, DEFAULT, HARD, ON, LINEAR)


#%% contact property

class ContactProperty(object):

    def __init__(self, name, behaviors, model=None):
        '''
        Parameters
        ----------
        name : str
        behaviors : array-like of contact behaviour instances.
        model : abaqus mdb object

        Notes
        -----
        -for further informations see p25-54 of [1].
        '''
        self.name = name
        self.behaviors = behaviors
        # computations
        if model:
            self.create_contact_property(model)

    def create_contact_property(self, model):

        # create abaqus object
        contact = model.ContactProperty(self.name)

        # add behaviours
        for behavior in self.behaviors:
            add_behavior = getattr(contact, behavior.method_name)
            add_behavior(**behavior.args)


#%% contact behaviours

class TangentialBehavior(object):

    method_name = 'TangentialBehavior'

    def __init__(self, formulation=FRICTIONLESS, directionality=ISOTROPIC,
                 slipRateDependency=OFF, pressureDependency=OFF,
                 temperatureDependency=OFF, dependencies=0,
                 exponentialDecayDefinition=COEFFICIENTS, table=(),
                 shearStressLimit=None, maximumElasticSlip=FRACTION, fraction=0.,
                 absoluteDistance=0., elasticSlipStiffness=None,
                 nStateDependentVars=0, useProperties=OFF):
        '''
        Parameters
        ----------
        formulation : abaqus constant
            Friction formulation.
        directionality : abaqus constant
            Directionality of the friction.
        slipRateDependency, pressureDependency, temperatureDependency : bool
            Whether the data depend on the slip rate, contact pressure,
            temperature.
        dependencies : int
            Number of field variables.
        exponentialDecayDefinition : abaqus constant
        table : sequence of sequences of floats
            Specification of tangential behavior.
        shearStressLimit : float
        maximumElasticSlip : abaqus constant
        fraction : float
            Fraction of a characteristic surface dimension.
        absoluteDistance : float
        elasticSlipStiffness : float
        nStateDependentVars : int
            Number of state-dependent variables.
        useProperties : bool
            Whether property values will be used.

        Notes
        -----
        -for further informations see p25-65 of [1].
        '''
        # create args dict
        self.args = {'formulation': formulation,
                     'directionality': directionality,
                     'slipRateDependency': slipRateDependency,
                     'pressureDependency': pressureDependency,
                     'temperatureDependency': temperatureDependency,
                     'dependencies': dependencies,
                     'exponentialDecayDefinition': exponentialDecayDefinition,
                     'table': table,
                     'shearStressLimit': shearStressLimit,
                     'maximumElasticSlip': maximumElasticSlip,
                     'fraction': fraction,
                     'absoluteDistance': absoluteDistance,
                     'elasticSlipStiffness': elasticSlipStiffness,
                     'nStateDependentVars': nStateDependentVars,
                     'useProperties': useProperties}


class NormalBehavior(object):

    method_name = 'NormalBehavior'

    def __init__(self, contactStiffness=0., pressureOverclosure=HARD,
                 allowSeparation=ON, maxStiffness=None, table=(),
                 constraintEnforcementMethod=DEFAULT, overclosureFactor=0.,
                 overclosureMeasure=0., contactStiffnessScaleFactor=1.,
                 initialStiffnessScaleFactor=1., clearanceAtZeroContactPressure=0.,
                 stiffnessBehavior=LINEAR, stiffnessRatio=.01, upperQuadraticFactor=.03,
                 lowerQuadraticRatio=.33333):
        '''
        Parameters
        ----------
        contactStiffness : abaqus constant or float
            Contact stiffness. Only applicable when pressureOverclosure=LINEAR
            or pressureOverclosure=HARD and constraintEnforcementMethod=LAGRANGE
            or PENALTY.
        pressureOverclosure : abaqus constant
            Pressure-overclosure relationship to be used.
        allowSeparation : bool
            Whether to allow separation after contact.
        maxStiffness : float
        table : sequence of sequences of floats
            Normal behavior properties. Only applicable when
            pressureOverclosure=EXPONENTIAL or TABULAR.
        constraintEnforcementMethod : abaqus constant
            Method for enforcement of the contact contraints.
        overclosureFactor : float
            Overclosure measure (used to delineate the segments of the
            pressure-overclosure curve) as a percentage of the minumum
            element size in the contact region.
        overclosureMeasure : float
            Overclosure measure (used to delineate the segments of the
            pressure-overclosure curve) directly.
        contactStiffnessScaleFactor : float
            Penalty stiffness or geometric scaling of the 'base' stiffness.
        initialStiffnessScaleFactor : float
            Additional scale factor for the 'base' default stiffness.
        clearanceAtZeroContactPressure : float
            The clearance at which the contact pressure is zero.
        stiffnessBehavior : abaqus constant
            Type of penalty stiffness. Only applicable when
            constraintEnforcementMethod=PENALTY.
        stiffnessRatio : float
            Ratio of the initial stiffness divided by the final stiffness. Only
            applicable when stiffnessBehavior=NONLINEAR.
        upperQuadraticFactor : float
            Ratio of the overclosure at the maximum stiffness divided by the
            characteristic facet length. Only applicable when
            stiffnessBehavior=NONLINEAR.
        lowerQuadraticRatio : float
            Ratio of the overclosure at the initial stiffness divided by the
            overclosure at the maximum stiffness, both relative to the clearance
            at which the contact pressure is zero. Only applicable when
            stiffnessBehavior=NONLINEAR.

        Notes
        -----
        -for further informations see p25-127 of [1].
        '''
        # create args dict
        self.args = {'contactStiffness': contactStiffness,
                     'pressureOverclosure': pressureOverclosure,
                     'allowSeparation': allowSeparation,
                     'maxStiffness': maxStiffness,
                     'table': table,
                     'constraintEnforcementMethod': constraintEnforcementMethod,
                     'overclosureFactor': overclosureFactor,
                     'overclosureMeasure': overclosureMeasure,
                     'contactStiffnessScaleFactor': contactStiffnessScaleFactor,
                     'initialStiffnessScaleFactor': initialStiffnessScaleFactor,
                     'clearanceAtZeroContactPressure': clearanceAtZeroContactPressure,
                     'stiffnessBehavior': stiffnessBehavior,
                     'stiffnessRatio': stiffnessRatio,
                     'upperQuadraticFactor': upperQuadraticFactor,
                     'lowerQuadraticRatio': lowerQuadraticRatio}


class GeometricProperties(object):

    method_name = 'GeometricProperties'

    def __init__(self, contactArea=1., padThickness=None):
        '''
        Parameters
        ----------
        contactArea : float
            Out-of-plane thickness of the surface for a two-dimensional model
            or cross-sectional area for every node in the node-based surface.
        padThickness : float
            Thickness of an interfacial layer between contacting surfaces.

        Notes
        -----
        -for further informations see p25-106 of [1].
        '''
        # create args dict
        self.args = {'contactArea': contactArea,
                     'padThickness': padThickness}
