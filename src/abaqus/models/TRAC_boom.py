'''
Created on 2020-04-15 11:26:39
Last modified on 2020-04-22 16:06:45
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Modelling TRAC boom.


References
----------
1. Bessa, M. A. and S. Pellegrino (2018) International Journal of Solids and
Structures 139-140: 174-188.
'''


#%% imports

# abaqus library
from abaqus import session
from abaqusConstants import (BUCKLING_MODES, ON)

# third-party
import numpy as np

# local library
from ..geometry.structures import TRACBoom
from ..material.abaqus_materials import LaminaMaterial
from ..modelling.model import BasicModel
from ..modelling.step import BuckleStep
from ..modelling.step import StaticRiksStep
from ..modelling.bcs import DisplacementBC
from ..modelling.bcs import Moment
from ..modelling.misc import AddToInp
from ..post_processing.get_data import get_ydata_from_nodeSets_field_output


#%% TRAC boom model

class TRACBoomModel(BasicModel):

    def __init__(self, name, sim_type, job_name, height, radius, theta,
                 thickness, length, material_name, layup=None, rotation_axis=1,
                 job_description='', previous_model=None,
                 previous_model_results=None):
        '''
        Parameters
        ----------
        name : str
        sim_type : str
            Type of simulation. Possible valures are 'lin_buckle', 'freq' and
            'riks'.
        job_name : str
        height, radius, theta, thickness, lenght : float
            TRAC boom geometric parameters.
        material_name : str
            TRAC boom material name.
        layup : array
            TRAC boom layup if composite material.
        rotation_axis : int. possible values = 1 or 2
            Axis about which moment will be applied.
        job_description : str
        previous_model : TRACBoomModel instance
            Model from which extract boundary conditions information. Only
            required if sim_type='riks' and previous_model_results is None. # TODO: also add static
        previous_model_results : dict
            Results of previous model with format given by perform_post_processing.


        Notes
        -----
        -some simulation types (e.g. Riks) must be preceded by other
        simulations.
        '''
        # initialize parent
        BasicModel.__init__(self, name, job_name, job_description)
        # specific variables
        self.applied_moment = 2.
        # TODO: review how these values are computed
        self.mode_amplitudes = [7.88810e-04, 8.15080e-04]
        # store variables
        self.sim_type = sim_type
        self.height = height
        self.radius = radius
        self.theta = theta
        self.thickness = thickness
        self.length = length
        self.material_name = material_name
        self.layup = layup
        self.rotation_axis = rotation_axis
        self.previous_model = previous_model
        self.previous_model_results = previous_model_results

    def assemble_puzzle(self):

        # TODO: extend to different materials (geometry already prepared)

        # set materials
        material = LaminaMaterial(self.material_name, create_section=False)
        self.materials.append(material)

        # define objects
        # set TRAC boom
        trac_boom = TRACBoom(self.height, self.radius, self.theta,
                             self.thickness, self.length, material,
                             layup=self.layup, rotation_axis=self.rotation_axis,
                             name='TRACBOOM')
        self.geometry_objects.append(trac_boom)

        # set step
        step = self._create_step()
        self.steps.append(step)

        # apply boundary conditions
        bcs = self._apply_bcs(trac_boom, step.name)
        self.bcs.extend(bcs)

        # add text to inp
        self.inp_additions.append(self._create_inp_additions())

    def perform_post_processing(self, *args):
        fnc = getattr(self, '_perform_post_processing_%s' % self.sim_type)
        return fnc(*args)

    def _create_step(self):
        fnc = getattr(self, '_create_step_%s' % self.sim_type)
        return fnc()

    def _create_step_lin_buckle(self):
        step_name = 'BUCKLE_STEP'
        buckle_step = BuckleStep(step_name, minEigen=0.)

        return buckle_step

    def _create_step_riks(self):
        step_name = 'RIKS_STEP'
        riks_step = StaticRiksStep(step_name, nlgeom=ON, maxNumInc=200,
                                   initialArcInc=5e-2, maxArcInc=0.1)

        return riks_step

    def _apply_bcs(self, trac_boom, step_name):

        # displacement bcs
        fix_1, fix_2 = self._apply_disp_bcs(trac_boom, step_name)

        # load bcs
        moment = self._apply_load_bcs(trac_boom, step_name)

        return fix_1, fix_2, moment

    def _apply_disp_bcs(self, trac_boom, step_name):

        # initialization
        kwargs_disp = {'ur%i' % (1 + int(not (trac_boom.rotation_axis - 1))): 0}

        # fix ref point minus
        position = trac_boom.ref_point_positions[0]
        region_name = trac_boom._get_ref_point_name(position)
        fix_1 = DisplacementBC('BC_ZMINUS', createStepName=step_name,
                               region=region_name, u1=0., u2=0., u3=0., ur3=0,
                               buckleCase=BUCKLING_MODES, **kwargs_disp)

        # fix ref point plus
        position = trac_boom.ref_point_positions[-1]
        region_name = trac_boom._get_ref_point_name(position)
        fix_2 = DisplacementBC('BC_ZPLUS', createStepName=step_name,
                               region=region_name, u1=0., u2=0., ur3=0,
                               buckleCase=BUCKLING_MODES, **kwargs_disp)

        return fix_1, fix_2

    def _apply_load_bcs(self, trac_boom, step_name):

        # get moment value
        moment = self.applied_moment
        if self.sim_type == 'riks':
            moment *= self._get_moment_for_riks()

        # apply moment
        kwargs_moment = {'cm%i' % trac_boom.rotation_axis: moment}
        position = trac_boom.ref_point_positions[-1]
        region_name = trac_boom._get_ref_point_name(position)
        moment = Moment('APPLIED_MOMENT', createStepName=step_name,
                        region=region_name, **kwargs_moment)

        return moment

    def _get_moment_for_riks(self):
        '''
        Notes
        -----
        -Assumes odb of the previous simulation is available in the working
        directory.
        '''

        # access odb
        odb_name = '%s.odb' % self.previous_model.job_name
        odb = session.openOdb(name=odb_name)

        # get results
        if self.previous_model_results is None:
            self.previous_model_results = self.previous_model.perform_post_processing(odb)

        return self.previous_model_results['critical_moment']

    def _create_inp_additions(self):
        fnc = getattr(self, '_create_inp_additions_%s' % self.sim_type)
        return fnc()

    def _create_inp_additions_lin_buckle(self):
        text = ['*NODE FILE, frequency=1', 'U']
        return AddToInp(text, self.job_name, section='OUTPUT REQUESTS')

    def _create_inp_additions_riks(self):

        # initialization
        amp_factors = [amp / max_disp for amp, max_disp in
                       zip(self.mode_amplitudes, self.previous_model_results['max_disp'][1:3])]
        text = ['*IMPERFECTION, FILE=DoE1_linear_buckle, STEP=1\n']

        # add imperfection values
        for i, amp_factor in enumerate(amp_factors):
            text.append('%i, %f\n' % (i + 1, amp_factor))

        return AddToInp(text, self.job_name, section='BOUNDARY CONDITIONS')

    def _perform_post_processing_lin_buckle(self, odb):
        '''
        Parameters
        ----------
        odb : Abaqus odb object
        '''

        # initialization
        step = odb.steps[odb.steps.keys()[-1]]
        frames = step.frames

        # get maximum displacements
        variable = 'U'
        directions = [1, 2, 3]
        nodeSets = [odb.rootAssembly.nodeSets[' ALL NODES']]
        values = get_ydata_from_nodeSets_field_output(odb, nodeSets, variable,
                                                      directions=directions,
                                                      frames=frames)
        max_disps = []
        for value in values:
            max_disp = np.max(np.abs(np.array(value)))
            max_disps.append(max_disp)

        # get eigenvalue
        eigenvalue = float(frames[1].description.split('EigenValue =')[1])

        # write output
        data = {'max_disp': max_disps,
                'critical_moment': eigenvalue}

        return data
