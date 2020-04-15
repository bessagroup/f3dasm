'''
Created on 2020-04-15 11:26:39
Last modified on 2020-04-15 19:16:54
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
from abaqusConstants import BUCKLING_MODES

# third-party
import numpy as np

# local library
from ..geometry.structures import TRACBoom
from ..material.abaqus_materials import LaminaMaterial
from ..modelling.model import BasicModel
from ..modelling.step import BuckleStep
from ..modelling.bcs import DisplacementBC
from ..modelling.bcs import Moment
from ..modelling.misc import AddToInp
from ..post_processing.get_data import get_ydata_from_nodeSets_field_output


#%% TRAC boom model

# TODO: assemble puzzle will have to be automatically performed when init


class TRACBoomModel(BasicModel):

    def __init__(self, name, job_name, job_description=''):
        # initialize parent
        BasicModel.__init__(self, name, job_name, job_description)
        # specific variables
        self.applied_moment = 2.

    def _apply_bcs(self, trac_boom, step_name):
        # TODO: bcs change regarding the type of simulation

        # initialization
        kwargs_disp = {'ur%i' % (1 + int(not (trac_boom.rotation_axis - 1))): 0}
        kwargs_moment = {'cm%i' % trac_boom.rotation_axis: self.applied_moment}

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

        # apply moment
        position = trac_boom.ref_point_positions[-1]
        region_name = trac_boom._get_ref_point_name(position)
        moment = Moment('APPLIED_MOMENT', createStepName=step_name,
                        region=region_name, **kwargs_moment)

        return fix_1, fix_2, moment

    def assemble_puzzle(self, height, radius, theta, thickness, length,
                        material_name, layup=None, rotation_axis=1):

        # TODO: extend to different materials (geometry already prepared)

        # set materials
        material = LaminaMaterial(material_name, create_section=False)
        self.materials.append(material)

        # define objects
        # set TRAC boom
        trac_boom = TRACBoom(height, radius, theta, thickness, length, material,
                             layup=layup, rotation_axis=rotation_axis,
                             name='TRACBOOM')
        self.geometry_objects.append(trac_boom)

        # set step
        step_name = 'BUCKLE_STEP'
        buckle_step = BuckleStep(step_name, minEigen=0.)
        self.steps.append(buckle_step)

        # apply boundary conditions
        bcs = self._apply_bcs(trac_boom, step_name)
        self.bcs.extend(bcs)

        # add text to inp
        text = ['*NODE FILE, frequency=1', 'U']
        self.inp_additions.append(AddToInp(text, self.job_name, section='OUTPUT REQUESTS'))

    def post_processing(self, odb):
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
