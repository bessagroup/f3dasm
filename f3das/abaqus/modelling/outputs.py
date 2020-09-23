'''
Created on 2020-04-15 18:00:10
Last modified on 2020-09-22 14:45:29
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Request outputs for abaqus simulations.


References
----------
1. Simulia (2015). ABAQUS 2016: Scripting Reference Guide
'''

# TODO: understand how to deal with steValuesInStep


#%% imports

# abaqus
from abaqusConstants import (MODEL, PRESELECT)

# standard library
import abc


#%% abstract classes

class Output(object):
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
            self.create_output(model)

    def _get_set(self, set_name, model):

        set_name_split = set_name.split('.')
        if len(set_name_split) > 1:
            instance_name, set_name = set_name_split
            abaqusSet = model.rootAssembly.instances[instance_name].sets[set_name]
        else:
            abaqusSet = model.rootAssembly.sets[set_name]

        return abaqusSet

    def create_output(self, model):

        # get method
        create_output = getattr(model, self.method_name)

        # get sets
        args = {}
        if hasattr(self, 'sets'):
            for key, value in self.sets.items():
                if type(value) is str:
                    surf = self._get_set(value, model)
                elif callable(value):
                    surf = value(model)
                else:
                    surf = value
                args[key] = surf
        args.update(self.args)

        # create output
        create_output(name=self.name,
                      createStepName=self.createStepName, **args)


#%% field outputs

class FieldOutputRequest(Output):

    method_name = 'FieldOutputRequest'

    def __init__(self, name, createStepName, model=None, region=MODEL,
                 variables=PRESELECT, **kwargs):
        '''
        Parameters
        ----------
        region : abaqus constant or abaqus Set object or str
            Region from which output is requested.
        variables : sequence of str
            Output request variable or component names.
        frequency : abaqus constant or int
            Output frequency in increments.
        modes : abaqus constant or sequence of int
            List of eigenmodes for which output is desired.
        timeInterval : abaqus constant or float
            Time interval at which the output states are to be written.
        numIntervals : int
            Number of intervals at which output database states are to be
            written.
        timeMarks : bool
            When to wite the resuts to the output database.
        boltLoad : str
            Bolt load from which output is requested.
        sectionPoints : abaqus constant or sequence of int
            Section points for which output is requested.
        interactions : sequence of str
            Interaction names.
        rebar : abaqus constant
            Whether output is requested for rebar.
        filter : abaqus constant or str
            Name of an output filter object.
        directions : bool
            Whether to output directions of the local material coordinate system.
        fasteners : str
            Fastener name.
        assembledFastener : str
            Assembled fastener name.
        assembledFastenerSet : str
            Set name from the model referenced by the assembled fastener.
        exteriorOnly : bool
            Whether the output domain is restricted to the exterior of the model.
        layupNames : sequence of str
            Composite layer names.
        layupLocationMethod : abaqus constat
            Output location for composite layups.
        outputAtPlyTop, outputAtPlyMid, outputAtPlyBottom : bool
            Whether to output at ply top/mid/bottom section point.

        Notes
        -----
        -for further informations see p51-3 of [1].
        -not all possible arguments are explicitly defined because they
        depend on the step type and the keywords are not recognized if
        they are not relevant.
        '''
        # store variables
        self.name = name
        self.createStepName = createStepName
        # create args dict
        self.sets = {'region': region}
        self.args = {'variables': variables}
        self.args.update(kwargs)
        # initialize parent
        Output.__init__(self, name, createStepName, model=model)


#%% history outputs

class HistoryOutputRequest(Output):

    method_name = 'HistoryOutputRequest'

    def __init__(self, name, createStepName, model=None, region=MODEL,
                 variables=PRESELECT, **kwargs):
        '''
        Parameters
        ----------
        region: abaqus constant or abaqus Set object or str
            Region from which output is requested.
        variables: sequence of str
            Output request variable or component names.
        frequency: abaqus constant or int
            Output frequency in increments.
        modes: abaqus constant or sequence of int
            List of eigenmodes for which output is desired.
        timeInterval: abaqus constant or float
            Time interval at which the output states are to be written.
        numIntervals: int
            Number of intervals at which output database states are to be
            written.
        boltLoad: str
            Bolt load from which output is requested.
        sectionPoints: abaqus constant or sequence of int
            Section points for which output is requested.
        stepName: str
        interactions: sequence of str
            Interaction names.
        contourIntegral: str
            Contour integral name.
        numberOfContours: int
            Number of contour integrals to output for the contour integral
            object.
        stressInitializationStep: str
        contourType: abaqus constant
            Type of contour integral.
        kFactorDirection: abaqus constant
            Stress intensity factor direction. Only applicable when
            contourType = K_FACTORS.
        rebar: abaqus constant
            Whether output is requested for rebar.
        integratedOutputSection: str
        springs: sequence of str
            Springs / dashpots names.
        filter: abaqus constant or str
            Name of an output filter object.
        fasteners: str
            Fastener name.
        assembledFastener: str
            Assembled fastener name.
        assembledFastenerSet: str
            Set name from the model referenced by the assembled fastener.
        sensor: bool
            Whether to associate the output request with a sensor definition.
        useGlobal: bool
            Whether to output vector - valued nodal variables in the global
            directions.

        Notes
        -----
        -for further informations see p51 - 12 of[1].
        -not all possible arguments are explicitly defined because they
        depend on the step type and the keywords are not recognized if
        they are not relevant.
        '''
        # create args dict
        self.sets = {'region': region}
        self.args = {'variables': variables}
        self.args.update(kwargs)
        # initialize parent
        Output.__init__(self, name, createStepName, model=model)
