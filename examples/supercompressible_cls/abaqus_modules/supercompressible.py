'''
Created on 2020-04-20 19:18:16
Last modified on 2020-11-17 10:49:25

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Modelling supercompressible metamaterial.
'''


# imports

from __future__ import division

# abaqus library
from abaqus import session
from abaqusConstants import (BUCKLING_MODES, ON, HARD, OFF, FINITE)

# standard library
from abc import ABCMeta
from abc import abstractmethod

# third-party
import numpy as np
from f3dasm.abaqus.geometry.structures import Supercompressible
from f3dasm.abaqus.modelling.model import GenericModel
from f3dasm.abaqus.modelling.step import BuckleStep
from f3dasm.abaqus.modelling.step import StaticRiksStep
from f3dasm.abaqus.modelling.bcs import DisplacementBC
from f3dasm.abaqus.modelling.bcs import ConcentratedForce
from f3dasm.abaqus.modelling.utils import InpAdditions
from f3dasm.abaqus.modelling.interaction_properties import ContactProperty
from f3dasm.abaqus.modelling.interaction_properties import NormalBehavior
from f3dasm.abaqus.modelling.interaction_properties import GeometricProperties
from f3dasm.abaqus.modelling.interactions import SurfaceToSurfaceContactStd
from f3dasm.abaqus.modelling.outputs import HistoryOutputRequest
from f3dasm.abaqus.post_processing.get_data import get_ydata_from_nodeSets_field_output
from f3dasm.abaqus.post_processing.get_data import get_xydata_from_nodes_history_output
from f3dasm.abaqus.post_processing.get_data import get_eigenvalues
from f3dasm.abaqus.post_processing.nodes_and_elements import get_nodes_given_set_names


# supercompressible metamaterial

# TODO: create a model common to TRAC boom (it is the same strategy); ImperfectionModel
# TODO: simplify geometry generation
# TODO: delete dependency on geometry objects


class SupercompressibleModel(GenericModel):
    __metaclass__ = ABCMeta

    def __init__(self, name, job_info, geometry_info):
        # initialize parent
        super(SupercompressibleModel, self).__init__(name, job_info)
        # store variables
        self.geometry_info = geometry_info
        # auxiliar variables
        self.supercompressible = None

    @abstractmethod
    def perform_post_processing(self, odb):
        pass

    def set_geometries(self):
        # TODO: pass a material in
        self.supercompressible = Supercompressible(**self.geometry_info)

        return self.supercompressible

    def set_bcs(self):

        # initialization
        bcs = []
        step_name = self.step_info['name']

        # displacement bcs
        bcs.extend(self._set_disp_bcs(step_name))

        # load bcs
        if self.sim_type == 'lin_buckle':
            bcs.extend(self._set_load_bcs(step_name))

        return bcs

    def _set_disp_bcs(self, step_name):

        # initialization
        disp_bcs = []

        # fix ref point minus
        position = self.supercompressible.ref_point_positions[0]
        region_name = self.supercompressible._get_ref_point_name(position)
        disp_bcs.append(DisplacementBC('BC_FIX', createStepName=step_name,
                                       region=region_name, u1=0., u2=0., u3=0.,
                                       ur1=0., ur2=0., ur3=0., buckleCase=BUCKLING_MODES))

        return disp_bcs


class SupercompressibleLinearBuckleModel(SupercompressibleModel):

    sim_type = 'lin_buckle'

    def __init__(self, name, job_info, geometry_info):
        super(SupercompressibleLinearBuckleModel, self).__init__(
            name, job_info, geometry_info)
        self.step_info = {'name': 'BUCKLE_STEP',
                          'minEigen': 0.}
        # specific variables
        self.applied_load = -1.

    def set_steps(self):
        return BuckleStep(**self.step_info)

    def perform_post_processing(self, odb):
        # TODO: simplify for readability

        # initialization
        step = odb.steps[odb.steps.keys()[-1]]
        frames = step.frames

        # get maximum displacements
        variable = 'UR'
        directions = (1, 2, 3)
        nodeSet = odb.rootAssembly.nodeSets[' ALL NODES']
        values = get_ydata_from_nodeSets_field_output(odb, nodeSet, variable,
                                                      directions=directions,
                                                      frames=frames)

        max_disps = []
        for value in values:
            max_disp = np.max(np.abs(np.array(value)))
            max_disps.append(max_disp)

        # get eigenvalue
        eigenvalues = get_eigenvalues(odb, frames=frames)

        # get top ref point info
        position = self.supercompressible.ref_point_positions[-1]
        ztop_set_name = self.supercompressible._get_ref_point_name(position)
        ztop_set = odb.rootAssembly.nodeSets[ztop_set_name]

        # is coilable?
        ztop_ur = get_ydata_from_nodeSets_field_output(odb, ztop_set, 'UR',
                                                       directions=(3,),
                                                       frames=list(frames)[1:])
        ztop_u = get_ydata_from_nodeSets_field_output(odb, ztop_set, 'U',
                                                      directions=(1, 2,),
                                                      frames=list(frames)[1:])
        ur = ztop_ur[0]
        u = ztop_u[0]
        coilable = int(abs(ur[0][0]) > 1.0e-4 and abs(u[0][0]) < 1.0e-4 and abs(u[1][0]) < 1.0e-4)

        # write output
        data = {'max_disps': max_disps,
                'loads': eigenvalues,
                'coilable': coilable}

        return data

    def set_inp_additions(self):
        text = ['*NODE FILE, frequency=1', 'U']
        return InpAdditions(text, self.job_info['name'], section='OUTPUT REQUESTS')

    def _set_load_bcs(self, step_name):

        # apply moment
        position = self.supercompressible.ref_point_positions[-1]
        region_name = self.supercompressible._get_ref_point_name(position)
        moment = ConcentratedForce('APPLIED_FORCE', createStepName=step_name,
                                   region=region_name, cf3=self.applied_load)

        return [moment]


class SupercompressibleRiksModel(SupercompressibleModel):
    sim_type = 'riks'

    def __init__(self, name, job_info, geometry_info, previous_model,
                 previous_model_results, imperfection):
        super(SupercompressibleRiksModel, self).__init__(
            name, job_info, geometry_info)
        self.step_info = {'name': 'RIKS_STEP',
                          'nlgeom': ON,
                          'maxNumInc': 400,
                          'initialArcInc': 5e-2,
                          'maxArcInc': 0.5}
        # store variables
        self.previous_model = previous_model
        self.previous_model_results = previous_model_results
        self.mode_amplitudes = [imperfection]

    def _must_abort(self):

        # verify coilability
        if self.sim_type == 'riks':
            coilable = self._verify_coilability()

            if not coilable:
                return True

    def set_steps(self):
        return StaticRiksStep(**self.step_info)

    def perform_post_processing(self, odb):

        # initialization
        data = {}

        # reference point data
        position = self.supercompressible.ref_point_positions[-1]
        variables = ['U', 'UR', 'RF', 'RM']
        set_names = [self.upercompressible._get_ref_point_name(position)]
        nodes = get_nodes_given_set_names(odb, set_names)
        # get variables
        for variable in variables:
            data[variable] = get_xydata_from_nodes_history_output(
                odb, nodes, variable)[0]

        # add information if not generalized section
        if self.supercompressible.geometry_info.cross_section_props.get('type', 'generalized') != 'generalized':
            step = odb.steps[odb.steps.keys()[-1]]
            frames = step.frames
            elemSet = odb.rootAssembly.elementSets[' ALL ELEMENTS']
            data['E'] = get_ydata_from_nodeSets_field_output(odb, elemSet, 'E',
                                                             directions=(1, 3,),
                                                             frames=frames)

        return data

    def set_outputs(self):

        # initialization
        step_name = self.step_info['name']

        # energy outputs
        history_outputs = [HistoryOutputRequest(
            name='ENERGIES', createStepName=step_name, variables=('ALLEN',))]

        # load-disp outputs
        position = self.supercompressible.ref_point_positions[-1]
        region = self.supercompressible._get_ref_point_name(position)
        name = 'RP_%s' % position
        history_outputs.append(HistoryOutputRequest(
            name=name, createStepName=step_name,
            region=region, variables=('U', 'RF')))

        return history_outputs

    def set_inp_additions(self):

        # initialization
        previous_file_name = self.previous_model.job_info['name']
        text = ['*IMPERFECTION, FILE={}, STEP=1'.format(previous_file_name)]

        # get amplification factors
        max_disps = self._get_disps_for_riks()
        amp_factors = []
        for amp, max_disp in zip(self.mode_amplitudes, max_disps[1:]):
            amp_factors.append(amp / max_disp)

        # add imperfection values
        for i, amp_factor in enumerate(amp_factors):
            text.append('{}, {}'.format(i + 1, amp_factor))

        return InpAdditions(text, self.job_info['name'], section='INTERACTIONS')

    def _get_disps_for_riks(self):

        # get results
        return self.previous_model_results['max_disps']

    def set_interactions(self):

        # add contact properties
        # contact behaviour
        normal_beh = NormalBehavior(allowSeparation=OFF, pressureOverclosure=HARD)
        geom_prop = GeometricProperties(contactArea=1.)
        # contact property
        contact = ContactProperty('IMP_TARG', behaviors=[normal_beh, geom_prop])

        # create interaction
        master = '%s.%s' % (self.supercompressible.surface_name, self.supercompressible.surface_name)
        slave = '%s.ALL_LONGERONS_SURF' % self.supercompressible.longerons_name
        interaction = SurfaceToSurfaceContactStd(
            name='IMP_TARG', createStepName='Initial', master=master,
            slave=slave, sliding=FINITE, interactionProperty=contact.name,
            thickness=OFF)

        return contact, interaction

    def _verify_coilability(self):
        '''
        Notes
        -----
        -Assumes odb of the previous simulation is available in the working
        directory.
        '''
        # TODO: coilable as integer
        if self.previous_model_results is None:
            # access odb
            odb_name = '%s.odb' % self.previous_model.job_info['name']
            odb = session.openOdb(name=odb_name)
            self.previous_model_results = self.previous_model.perform_post_processing(odb)
            odb.close()

        return int(self.previous_model_results['coilable'])

    def _set_disp_bcs(self, step_name):

        # shared with linear buckle
        disp_bcs = super(SupercompressibleRiksModel, self)._set_disp_bcs(step_name)

        # additional bcs
        vert_disp = - self.pitch
        position = self.supercompressible.ref_point_positions[-1]
        region_name = self.supercompressible._get_ref_point_name(position)
        disp_bcs.append(DisplacementBC('DISPLACEMENT', createStepName=step_name,
                                       region=region_name, u3=vert_disp,
                                       buckleCase=BUCKLING_MODES))

        return disp_bcs
