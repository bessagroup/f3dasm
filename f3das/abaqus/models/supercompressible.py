'''
Created on 2020-04-20 19:18:16
Last modified on 2020-09-22 16:17:46
Python 2.7.16
v0.1

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

# third-party
import numpy as np

# local library
from ..geometry.structures import Supercompressible
from ..modelling.model import BasicModel
from ..modelling.step import BuckleStep
from ..modelling.step import StaticRiksStep
from ..modelling.bcs import DisplacementBC
from ..modelling.bcs import ConcentratedForce
from ..modelling.utils import AddToInp
from ..modelling.interaction_properties import ContactProperty
from ..modelling.interaction_properties import NormalBehavior
from ..modelling.interaction_properties import GeometricProperties
from ..modelling.interactions import SurfaceToSurfaceContactStd
from ..modelling.outputs import HistoryOutputRequest
from ..post_processing.get_data import get_ydata_from_nodeSets_field_output
from ..post_processing.get_data import get_xydata_from_nodes_history_output
from ..post_processing.get_data import get_eigenvalues
from ..post_processing.nodes_and_elements import get_nodes_given_set_names


# supercompressible metamaterial

# TODO: create a model common to TRAC boom (it is the same strategy); ImperfectionModel
# TODO: if not coilable, then do not perform riks?

class SupercompressibleModel(BasicModel):

    def __init__(self, name, job_info, sim_type, n_longerons,
                 bottom_diameter, top_diameter, pitch, young_modulus,
                 shear_modulus, cross_section_props, twist_angle=0.,
                 transition_length_ratio=1., n_storeys=1, z_spacing='uni',
                 power=1., previous_model=None,
                 previous_model_results=None, imperfection=None):
        # initialize parent
        BasicModel.__init__(self, name, job_info)
        # specific variables
        self.applied_load = -1.
        # store variables
        self.sim_type = sim_type
        self.n_longerons = n_longerons
        self.bottom_diameter = bottom_diameter
        self.top_diameter = top_diameter
        self.pitch = pitch
        self.young_modulus = young_modulus
        self.shear_modulus = shear_modulus
        self.cross_section_props = cross_section_props
        self.twist_angle = twist_angle
        self.transition_length_ratio = transition_length_ratio
        self.n_storeys = n_storeys
        self.z_spacing = z_spacing
        self.power = power
        self.previous_model = previous_model
        self.previous_model_results = previous_model_results
        self.mode_amplitudes = [imperfection]

    def perform_post_processing(self, odb):
        fnc = getattr(self, '_perform_post_processing_%s' % self.sim_type)
        return fnc(odb)

    def _assemble_puzzle(self):

        # create objects
        supercompressible = Supercompressible(
            self.n_longerons, self.bottom_diameter, self.top_diameter,
            self.pitch, self.young_modulus, self.shear_modulus,
            self.cross_section_props, twist_angle=self.twist_angle,
            transition_length_ratio=self.transition_length_ratio,
            n_storeys=self.n_storeys, z_spacing=self.z_spacing,
            power=self.power)
        self._update_list(self.geometry_objects, supercompressible)

        # set step
        step = self._set_step()
        self._update_list(self.steps, step)

        # apply boundary conditions
        bcs = self._set_bcs(step.name)
        self._update_list(self.bcs, bcs)

        # set contact
        contact, interaction = self._set_contact()
        self._update_list(self.contact_properties, contact)
        self._update_list(self.interactions, interaction)

        # create outputs
        outputs = self._set_outputs(step)
        self._update_list(self.output_requests, outputs)

        # add text to inp
        inp_additions = self._set_inp_additions()
        self._update_list(self.inp_additions, inp_additions)

    def _set_step(self):
        fnc = getattr(self, '_set_step_%s' % self.sim_type)
        return fnc()

    def _set_step_lin_buckle(self):
        step_name = 'BUCKLE_STEP'
        buckle_step = BuckleStep(step_name, minEigen=0.)

        return buckle_step

    def _set_step_riks(self):
        step_name = 'RIKS_STEP'
        riks_step = StaticRiksStep(step_name, nlgeom=ON, maxNumInc=400,
                                   initialArcInc=5e-2, maxArcInc=0.5)

        return riks_step

    def _set_bcs(self, step_name):

        # initialization
        bcs = []
        supercompressible = self.geometry_objects[0]

        # displacement bcs
        bcs.extend(self._set_disp_bcs(supercompressible, step_name))

        # load bcs
        bcs.extend(self._set_load_bcs(supercompressible, step_name))

        return bcs

    def _set_disp_bcs(self, supercompressible, step_name):

        # initialization
        disp_bcs = []

        # fix ref point minus
        position = supercompressible.ref_point_positions[0]
        region_name = supercompressible._get_ref_point_name(position)
        disp_bcs.append(DisplacementBC('BC_FIX', createStepName=step_name,
                                       region=region_name, u1=0., u2=0., u3=0.,
                                       ur1=0., ur2=0., ur3=0., buckleCase=BUCKLING_MODES))

        # displacement
        if self.sim_type == 'riks':
            vert_disp = - self.pitch
            position = supercompressible.ref_point_positions[-1]
            region_name = supercompressible._get_ref_point_name(position)
            disp_bcs.append(DisplacementBC('DISPLACEMENT', createStepName=step_name,
                                           region=region_name, u3=vert_disp,
                                           buckleCase=BUCKLING_MODES))

        return disp_bcs

    def _set_load_bcs(self, supercompressible, step_name):

        if self.sim_type != 'lin_buckle':
            return []

        # apply moment
        position = supercompressible.ref_point_positions[-1]
        region_name = supercompressible._get_ref_point_name(position)
        moment = ConcentratedForce('APPLIED_FORCE', createStepName=step_name,
                                   region=region_name, cf3=self.applied_load)

        return [moment]

    def _set_contact(self):

        if hasattr(self, '_set_contact_%s' % self.sim_type):
            fnc = getattr(self, '_set_contact_%s' % self.sim_type)
        else:
            return [[], []]
        return fnc()

    def _set_contact_riks(self):

        # initialization
        supercompressible = self.geometry_objects[0]

        # add contact properties
        # contact behaviour
        normal_beh = NormalBehavior(allowSeparation=OFF, pressureOverclosure=HARD)
        geom_prop = GeometricProperties(contactArea=1.)
        # contact property
        contact = ContactProperty('IMP_TARG', behaviors=[normal_beh, geom_prop])

        # create interaction
        master = '%s.%s' % (supercompressible.surface_name, supercompressible.surface_name)
        slave = '%s.ALL_LONGERONS_SURF' % supercompressible.longerons_name
        interaction = SurfaceToSurfaceContactStd(
            name='IMP_TARG', createStepName='Initial', master=master,
            slave=slave, sliding=FINITE, interactionProperty=contact.name,
            thickness=OFF)

        return contact, interaction

    def _set_outputs(self, *args):
        if hasattr(self, '_set_outputs_%s' % self.sim_type):
            fnc = getattr(self, '_set_outputs_%s' % self.sim_type)
        else:
            return []
        return fnc(*args)

    def _set_outputs_riks(self, step):

        # initialization
        supercompressible = self.geometry_objects[0]

        # energy outputs
        history_outputs = [HistoryOutputRequest(
            name='ENERGIES', createStepName=step.name, variables=('ALLEN',))]

        # load-disp outputs
        position = supercompressible.ref_point_positions[-1]
        region = supercompressible._get_ref_point_name(position)
        name = 'RP_%s' % position
        history_outputs.append(HistoryOutputRequest(
            name=name, createStepName=step.name,
            region=region, variables=('U', 'RF')))

        return history_outputs

    def _set_inp_additions(self):
        fnc = getattr(self, '_set_inp_additions_%s' % self.sim_type)
        return fnc()

    def _set_inp_additions_lin_buckle(self):
        text = ['*NODE FILE, frequency=1', 'U']
        return AddToInp(text, self.job_info['name'], section='OUTPUT REQUESTS')

    def _set_inp_additions_riks(self):

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

        return AddToInp(text, self.job_info['name'], section='INTERACTIONS')

    def _get_disps_for_riks(self):
        '''
        Notes
        -----
        -Assumes odb of the previous simulation is available in the working
        directory.
        '''

        # get results
        if self.previous_model_results is None:
            # access odb
            odb_name = '%s.odb' % self.previous_model.job_info['name']
            odb = session.openOdb(name=odb_name)
            self.previous_model_results = self.previous_model.perform_post_processing(odb)
            odb.close()

        return self.previous_model_results['max_disps']

    def _perform_post_processing_lin_buckle(self, odb):

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
        supercompressible = self.geometry_objects[0]
        position = supercompressible.ref_point_positions[-1]
        ztop_set_name = supercompressible._get_ref_point_name(position)
        ztop_set = odb.rootAssembly.nodeSets[ztop_set_name]

        # is coilable?
        ztop_ur = get_ydata_from_nodeSets_field_output(odb, ztop_set, 'UR',
                                                       directions=(3,),
                                                       frames=list(frames)[1:])
        ztop_u = get_ydata_from_nodeSets_field_output(odb, ztop_set, 'U',
                                                      directions=(1, 2,),
                                                      frames=list(frames)[1:])
        coilable = [int(abs(ur[0][0]) > 1.0e-4 and abs(u[0][0]) < 1.0e-4 and abs(u[1][0]) < 1.0e-4)
                    for ur, u in zip(ztop_ur, ztop_u)]

        # write output
        data = {'max_disps': max_disps,
                'loads': eigenvalues,
                'coilable': coilable}

        return data

    def _perform_post_processing_riks(self, odb):

        # initialization
        data = {}
        supercompressible = self.geometry_objects[0]

        # reference point data
        position = supercompressible.ref_point_positions[-1]
        variables = ['U', 'UR', 'RF', 'RM']
        set_names = [supercompressible._get_ref_point_name(position)]
        nodes = get_nodes_given_set_names(odb, set_names)
        # get variables
        for variable in variables:
            data[variable] = get_xydata_from_nodes_history_output(
                odb, nodes, variable)[0]

        # add information if not generalized section
        if supercompressible.cross_section_props.get('type', 'generalized') != 'generalized':
            step = odb.steps[odb.steps.keys()[-1]]
            frames = step.frames
            elemSet = odb.rootAssembly.elementSets[' ALL ELEMENTS']
            data['E'] = get_ydata_from_nodeSets_field_output(odb, elemSet, 'E',
                                                             directions=(1, 3,),
                                                             frames=frames)

        return data
