'''
Created on 2020-04-21 11:11:26
Last modified on 2020-09-22 14:35:15
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Define abaqus interactions.

References
----------
1. Simulia (2015). ABAQUS 2016: Scripting Reference Guide
'''


#%% imports

# abaqus
from abaqusConstants import (NONE, COMPUTED, SURFACE_TO_SURFACE, ON, OFF, OMIT,)

# standard library
import abc


#%% abstract classes

class Interaction(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, createStepName, model=None):
        '''
        Parameters
        ----------
        name : str
        createStepName : str
            Step in which the boundary condition is created.
        model : abaqus mdb object
        '''
        self.name = name
        self.createStepName = createStepName
        # computations
        if model:
            self.create_interaction(model)

    def _get_surface(self, surf_name, model):

        surf_name_split = surf_name.split('.')
        if len(surf_name_split) > 1:
            instance_name, surf_name = surf_name_split
            surf = model.rootAssembly.instances[instance_name].surfaces[surf_name]
        else:
            surf = model.rootAssembly.surfaces[surf_name]

        return surf

    def create_interaction(self, model):

        # get method
        create_interaction = getattr(model, self.method_name)

        # get surfaces
        args = {}
        if hasattr(self, 'surfaces'):
            for key, value in self.surfaces.items():
                if type(value) is str:
                    surf = self._get_surface(value, model)
                elif callable(value):
                    surf = value(model)
                else:
                    surf = value
                args[key] = surf
        args.update(self.args)

        # create interaction
        create_interaction(name=self.name, createStepName=self.createStepName,
                           **args)


#%% particular interactions definition

class SurfaceToSurfaceContactStd(Interaction):

    method_name = 'SurfaceToSurfaceContactStd'

    def __init__(self, name, createStepName, master, slave, sliding,
                 interactionProperty, model=None, interferenceType=NONE, overclosure=0.,
                 interferenceDirectionType=COMPUTED, direction=(),
                 amplitude=None, smooth=.2, hcrit=0., extensionZone=0.1,
                 adjustMethod=NONE, adjustTolerance=0., adjustSet=None,
                 enforcement=SURFACE_TO_SURFACE, thickness=ON,
                 contactControls='', tied=OFF, initialClearance=OMIT,
                 halfThreadAngle=None, pitch=None, majorBoltDiameter=COMPUTED,
                 meanBoltDiameter=COMPUTED, datumAxis=None, useReverseDatumAxis=OFF,
                 clearanceRegion=None, surfaceSmoothing=NONE, bondingSet=None):
        '''
        Parameters
        ----------
        name : str
        createStepName : str
            Step in which the boundary condition is created.
        master : str or method or abaqus Surface object
            Master surface.
        slave : str or method or abaqus Surface object
            Slave surface.
        sliding : abaqus constant
            Contact formulation.
        interactionProperty : str
            Contact property associated with interaction.
        interferenceType : abaqus constant
            Type of time-dependent allowabe interference for contact pairs
            and contact elements.
        overclosure : float
            Maximum overclosure distance allowed.
        interferenceDirectionType : abaqus constant
            Method used to determine the interface direction.
        direction : array of float. shape = [3]
            X-, Y-, Z-direction cosine of the interference direction vector.
            Only applicable when interferenceDirectionType=DIRECTION_COSINE.
        amplitude : str
            Amplitude curve that defines the magnitude of the prescribed
            interference.
        smooth : float
            Degree of smoothing used for deformable rigid or rigid master
            surfaces. Only applicable when enforcement=NODE_TO_SURFACE.
        hcrit : float
            Distance by which a slave node must penetrate the master surface
            before abaqus reduces the increment.
        extensionZone : float
            A fraction of the end segment or facet edge length by which the
            master surface is to be extended to avoid numerical round-off
            errors associated with contact modelling.
        adjustMethod : abaqus constant
        adjustTolerance : float
            Adjust tolerance.
        adjustSet : abaqus region object
            Region to which the adjustment is to be applied.
        enforcement : abaqus constant
            Discretization method.
        thickness : bool
            Whether shell/membrane element thickness is considered.
        contactControls : str
            Name of the ContactControl object associated with this interaction.
        tied : bool
            Whether surfaces are to be 'tied' together.
        initialClearance : abaqus constant or float
            Initial clearance at regions of contact.
        halfThreadAngle : sequence of floats
            Half thread angle used for bolt clearance.
        pitch : sequence of floats
            Pitch used for bolt clearance.
        majorBoltDiameter : abaqus constant or float
            Mean diameter of the bolt used for bolt clearance.
        datumAxis : datumAxis object
            Orientation of the bolt hole when specifying bolt clearance.
        useReverseDatumAxis : bool
            Whether to reverse the bolt clearance direction given by the datum
            axis.
        clearanceRegion : abaqus region object
            Contact region for which clearance is specified.
        surfaceSmoothing : abaqus contant
            Whether to use surface smoothing for geometric surfaces.
        bondingSet : abaqus region object
            Slave node subset for bonding. Only applicable when the contact
            property is CohesiveBehavior.

        Notes
        -----
        -for further informations see p25-177 of [1].
        '''
        # create args dict
        self.surfaces = {'master': master,
                         'slave': slave}
        self.args = {'sliding': sliding,
                     'interactionProperty': interactionProperty,
                     'interferenceType': interferenceType,
                     'overclosure': overclosure,
                     'interferenceDirectionType': interferenceDirectionType,
                     'direction': direction,
                     'amplitude': amplitude,
                     'smooth': smooth,
                     'hcrit': hcrit,
                     'extensionZone': extensionZone,
                     'adjustMethod': adjustMethod,
                     'adjustTolerance': adjustTolerance,
                     'enforcement': enforcement,
                     'thickness': thickness,
                     'contactControls': contactControls,
                     'tied': tied,
                     'initialClearance': initialClearance,
                     'halfThreadAngle': halfThreadAngle,
                     'pitch': pitch,
                     'majorBoltDiameter': majorBoltDiameter,
                     'meanBoltDiameter': meanBoltDiameter,
                     'datumAxis': datumAxis,
                     'useReverseDatumAxis': useReverseDatumAxis,
                     'clearanceRegion': clearanceRegion,
                     'surfaceSmoothing': surfaceSmoothing,
                     'bondingSet': bondingSet}
        if adjustSet is not None:
            self.args['adjustSet'] = adjustSet
        # initialize parent
        Interaction.__init__(self, name, createStepName, model=model)
