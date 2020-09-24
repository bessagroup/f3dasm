'''
Created on 2020-09-21 11:55:42
Last modified on 2020-09-22 07:57:56

@author: L. F. Pereira (lfpereira@fe.up.pt))

Main goal
---------
Show how the supercompressible example can be coded in a non object-oriented
strategy.
'''

# imports

# abaqus
from caeModules import *  # allow noGui
from abaqus import mdb, backwardCompatibility
from abaqusConstants import (THREE_D, DEFORMABLE_BODY, ON, OFF, STANDARD,
                             WHOLE_SURFACE, KINEMATIC, STANDALONE,
                             MIDDLE_SURFACE, FROM_SECTION,
                             CARTESIAN, IMPRINT, CONSTANT, BEFORE_ANALYSIS,
                             N1_COSINES, B31, FINER, ANALYTIC_RIGID_SURFACE,
                             NODAL)
import mesh

# standard library
import os
import sys
import re

# third-party
import numpy as np


# function definition

# TODO: create general supercompressible with if


def lin_buckle(name, job_name, n_longerons, bottom_diameter, top_diameter,
               pitch, young_modulus, shear_modulus, cross_section_props,
               twist_angle=0., transition_length_ratio=1., n_storeys=1,
               power=1., include_name='include_mesh', **kwargs):

    # create model
    model = mdb.Model(name=name)
    backwardCompatibility.setValues(reportDeprecated=False)
    if 'Model-1' in mdb.models.keys():
        del mdb.models['Model-1']

    # meshing
    cone_slope = (bottom_diameter - top_diameter) / bottom_diameter
    _meshing(model, n_longerons, bottom_diameter, pitch, cross_section_props,
             young_modulus, shear_modulus, cone_slope, include_name=include_name)

    # create linear buckling inp
    with open('{}.inp'.format(job_name), 'w') as File:
        File.write('** Include file with mesh of structure:\n')
        File.write('*INCLUDE, INPUT={}.inp\n'.format(include_name))
        File.write('** \n')
        File.write('** STEP: Step-1\n')
        File.write('** \n')
        File.write('*Step, name=Step-1\n')
        File.write('*Buckle, eigensolver=lanczos\n')
        File.write('20, 0., , , \n')
        File.write('** \n')
        File.write('** BOUNDARY CONDITIONS\n')
        File.write('** \n')
        File.write('** Name: BC_Zminus Type: Displacement/Rotation\n')
        File.write('*Boundary\n')
        File.write('RP_ZmYmXm, 1, 6\n')
        File.write('** \n')
        File.write('** LOADS\n')
        File.write('** \n')
        File.write('** Name: Applied_Moment   Type: Moment\n')
        File.write('*Cload\n')
        File.write('RP_ZpYmXm, 3, -1.00\n')
        File.write('** \n')
        File.write('*Node File\n')
        File.write('U \n')
        File.write('** \n')
        File.write('*EL PRINT,FREQUENCY=1\n')
        File.write('*NODE PRINT,FREQUENCY=1\n')
        File.write('*MODAL FILE\n')
        File.write('*OUTPUT,FIELD,VAR=PRESELECT\n')
        File.write('*OUTPUT,HISTORY,FREQUENCY=1\n')
        File.write('*MODAL OUTPUT\n')
        File.write('*End Step\n')


def post_process_lin_buckle(odb):
    # initialization
    match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?')

    # Determine the number of steps in the output database.
    mySteps = odb.steps
    stepKey = mySteps.keys()[0]
    step = mySteps[stepKey]
    numFrames = len(step.frames)
    #
    maxDisp = np.zeros((numFrames))
    RP_Zplus_nSet = odb.rootAssembly.nodeSets['RP_ZPYMXM']
    entireSTRUCTURE_nSet = odb.rootAssembly.nodeSets[' ALL NODES']
    #
    # Read critical buckling load
    P_crit = np.zeros(numFrames - 1)
    coilable = np.zeros(numFrames - 1)
    for iFrame_step in range(1, numFrames):
        # Read critical buckling load
        MODELframe = odb.steps[mySteps.keys()[0]].frames[iFrame_step]
        eigenValue = [float(x) for x in re.findall(match_number, MODELframe.description)]
        # eigenValue[1] is the eigenValue for Mode = eigenValue[0]
        P_crit[iFrame_step - 1] = eigenValue[1]
        # Now check if this is a coilable mode
        UR_Field = MODELframe.fieldOutputs['UR']
        UR_SubField = UR_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
        UR_Mode1_RP_ZpYmXm = UR_SubField.values[0].data
        U_Field = MODELframe.fieldOutputs['U']
        U_SubField = U_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
        U_Mode1_RP_ZpYmXm = U_SubField.values[0].data
        if (abs(UR_Mode1_RP_ZpYmXm[2]) > 1.0e-4) and (abs(U_Mode1_RP_ZpYmXm[0]) < 1.0e-4) and (abs(U_Mode1_RP_ZpYmXm[1]) < 1.0e-4):
            coilable[iFrame_step - 1] = 1
        else:
            coilable[iFrame_step - 1] = 0

    #
    UR_Field = MODELframe.fieldOutputs['UR']
    UR_SubField = UR_Field.getSubset(region=entireSTRUCTURE_nSet, position=NODAL)
    #
    if isinstance(UR_SubField.values[0].data, float):
        # Then variable is a scalar
        max_UR = np.zeros((numFrames))
    else:
        # Variable is an array
        max_UR = np.zeros((numFrames, len(UR_SubField.values[0].data)))

    for iFrame_step in range(numFrames):
        # Read frame
        MODELframe = odb.steps[mySteps.keys()[0]].frames[iFrame_step]
        # Variable: UR
        UR_Field = MODELframe.fieldOutputs['UR']
        UR_SubField = UR_Field.getSubset(region=entireSTRUCTURE_nSet, position=NODAL)
        #
        if isinstance(UR_SubField.values[0].data, float):
            # Then variable is a scalar:
            for strainValue in UR_SubField.values:
                if abs(strainValue.data) > abs(max_UR[iFrame_step]):
                    max_UR[iFrame_step] = abs(strainValue.data[0])
                #
        else:
            # Variable is an array:
            for strainValue in UR_SubField.values:
                for j in range(0, len(UR_SubField.values[0].data)):
                    if abs(strainValue.data[j]) > abs(max_UR[iFrame_step, j]):
                        max_UR[iFrame_step, j] = abs(strainValue.data[j])
                        max_UR[iFrame_step, j] = abs(strainValue.data[j])
        maxDisp[iFrame_step] = max(max_UR[iFrame_step, :])

    STRUCTURE_variables = {'loads': P_crit, 'max_disps': maxDisp,
                           'coilable': coilable}

    return STRUCTURE_variables


def riks(job_name, imperfection, previous_model_results,
         include_name='include_mesh', **kwargs):
    '''
    Notes
    -----
    1. Assumes linear buckling analyses was run previously.
    '''
    # get .fil
    for name in os.listdir('.'):
        if name.endswith('.fil'):
            fil_filename = name.split('.')[0]
            break

    # create riks inp
    with open('{}.inp'.format(job_name), 'w') as File:
        # if previous_model_results['coilable'][0] == 0:
        #     sys.exit()  # do not bother running the RIKS analysis because the material will not coil...
        # #
        File.write('** Include file with mesh of structure:\n')
        File.write('*INCLUDE, INPUT={}.inp\n'.format(include_name))
        File.write('** \n')
        File.write('** INTERACTION PROPERTIES\n')
        File.write('** \n')
        File.write('*SURFACE INTERACTION,NAME=IMP_TARG\n')
        File.write('1.,\n')
        File.write('*Surface Behavior, no separation, pressure-overclosure=HARD\n')
        File.write('***Surface Interaction, name=IntProp-1\n')
        File.write('** \n')
        File.write('** INTERACTIONS\n')
        File.write('** \n')
        File.write('***CONTACT PAIR,INTERACTION=IMP_TARG\n')
        File.write('**longerons-1-1.all_longerons_surface, AnalyticSurf-1-1.rigid_support\n')
        File.write('** Interaction: Int-1\n')
        File.write('*Contact Pair, interaction=IMP_TARG, type=SURFACE TO SURFACE, no thickness\n')
        File.write('longerons-1-1.all_longerons_surface, AnalyticSurf-1-1.rigid_support\n')
        File.write('**\n')
        File.write('** Seed an imperfection:\n')
        File.write('*IMPERFECTION, FILE={}, STEP=1\n'.format(fil_filename))
        mode_amplitude = imperfection / previous_model_results['max_disps'][1]
        File.write('1, ' + str(mode_amplitude) + '\n')
        File.write('** \n')
        File.write('** STEP: Step-1\n')
        File.write('** \n')
        File.write('*Step, name=Step-RIKS, nlgeom=YES, inc=400\n')
        File.write('*Static, riks\n')
        File.write('5.0e-2,1.0,,0.5\n')
        File.write('** \n')
        File.write('** BOUNDARY CONDITIONS\n')
        File.write('** \n')
        File.write('** Name: BC_Zminus Type: Displacement/Rotation\n')
        File.write('*Boundary\n')
        File.write('RP_ZmYmXm, 1, 6\n')
        File.write('** Name: BC_Zplus Type: Displacement/Rotation\n')
        File.write('*Boundary, type=displacement\n')
        File.write('RP_ZpYmXm, 3, 3, -1.23858e+02\n')
        File.write('** \n')
        File.write('** \n')
        File.write('** OUTPUT REQUESTS\n')
        File.write('** \n')
        File.write('** FIELD OUTPUT: F-Output-1\n')
        File.write('** \n')
        File.write('*Output, field, variable=PRESELECT, frequency=1\n')
        File.write('** \n')
        File.write('** HISTORY OUTPUT: H-Output-2\n')
        File.write('** \n')
        File.write('*Output, history, frequency=1\n')
        File.write('*Node Output, nset=RP_ZmYmXm\n')
        File.write('RF1, RF2, RF3, RM1, RM2, RM3, U1, U2\n')
        File.write('U3, UR1, UR2, UR3\n')
        File.write('** \n')
        File.write('** HISTORY OUTPUT: H-Output-3\n')
        File.write('** \n')
        File.write('*Node Output, nset=RP_ZpYmXm\n')
        File.write('RF1, RF2, RF3, RM1, RM2, RM3, U1, U2\n')
        File.write('U3, UR1, UR2, UR3\n')
        File.write('** \n')
        File.write('** HISTORY OUTPUT: H-Output-1\n')
        File.write('** \n')
        File.write('*Output, history, variable=PRESELECT, frequency=1\n')
        File.write('*End Step\n')


def post_process_riks(odb):
    # Determine the number of steps in the output database.
    mySteps = odb.steps
    numSteps = len(mySteps)
    #
    RP_Zplus_nSet = odb.rootAssembly.nodeSets['RP_ZPYMXM']
    RP_Zminus_nSet = odb.rootAssembly.nodeSets['RP_ZMYMXM']
    #
    # For each step, obtain the following:
    #     1) The step key.
    #     2) The number of frames in the step.
    #     3) The increment number of the last frame in the step.
    #
    totalNumFrames = 0
    for iStep in range(numSteps):
        stepKey = mySteps.keys()[iStep]
        step = mySteps[stepKey]
        numFrames = len(step.frames)
        totalNumFrames = totalNumFrames + numFrames
        #

    # Preallocate quantities for speed
    MODELframe = odb.steps[mySteps.keys()[0]].frames[0]  # Undeformed config.
    U_Field = MODELframe.fieldOutputs['U']
    U_RP_Zplus_SubField = U_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
    U_RP_Zminus_SubField = U_Field.getSubset(region=RP_Zminus_nSet, position=NODAL)
    #
    if isinstance(U_RP_Zplus_SubField.values[0].data, float):
        # Then variable is a scalar
        U_RP_Zplus = np.zeros((totalNumFrames))
        U_RP_Zminus = np.zeros((totalNumFrames))
    else:
        # Variable is an array
        U_RP_Zplus = np.zeros((totalNumFrames, len(U_RP_Zplus_SubField.values[0].data)))
        U_RP_Zminus = np.zeros((totalNumFrames, len(U_RP_Zminus_SubField.values[0].data)))

    UR_Field = MODELframe.fieldOutputs['UR']
    UR_RP_Zplus_SubField = UR_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
    UR_RP_Zminus_SubField = UR_Field.getSubset(region=RP_Zminus_nSet, position=NODAL)
    #
    if isinstance(UR_RP_Zplus_SubField.values[0].data, float):
        # Then variable is a scalar
        UR_RP_Zplus = np.zeros((totalNumFrames))
        UR_RP_Zminus = np.zeros((totalNumFrames))
    else:
        # Variable is an array
        UR_RP_Zplus = np.zeros((totalNumFrames, len(UR_RP_Zplus_SubField.values[0].data)))
        UR_RP_Zminus = np.zeros((totalNumFrames, len(UR_RP_Zminus_SubField.values[0].data)))

    RF_Field = MODELframe.fieldOutputs['RF']
    RF_RP_Zplus_SubField = RF_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
    RF_RP_Zminus_SubField = RF_Field.getSubset(region=RP_Zminus_nSet, position=NODAL)
    #
    if isinstance(RF_RP_Zplus_SubField.values[0].data, float):
        # Then variable is a scalar
        RF_RP_Zplus = np.zeros((totalNumFrames))
        RF_RP_Zminus = np.zeros((totalNumFrames))
    else:
        # Variable is an array
        RF_RP_Zplus = np.zeros((totalNumFrames, len(RF_RP_Zplus_SubField.values[0].data)))
        RF_RP_Zminus = np.zeros((totalNumFrames, len(RF_RP_Zminus_SubField.values[0].data)))

    RM_Field = MODELframe.fieldOutputs['RM']
    RM_RP_Zplus_SubField = RM_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
    RM_RP_Zminus_SubField = RM_Field.getSubset(region=RP_Zminus_nSet, position=NODAL)
    #
    if isinstance(RM_RP_Zplus_SubField.values[0].data, float):
        # Then variable is a scalar
        RM_RP_Zplus = np.zeros((totalNumFrames))
        RM_RP_Zminus = np.zeros((totalNumFrames))
    else:
        # Variable is an array
        RM_RP_Zplus = np.zeros((totalNumFrames, len(RM_RP_Zplus_SubField.values[0].data)))
        RM_RP_Zminus = np.zeros((totalNumFrames, len(RM_RP_Zminus_SubField.values[0].data)))

    # Loop over the Frames of this Step to extract values of variables
    numFrames = 0
    stepKey = mySteps.keys()[0]
    step = mySteps[stepKey]
    numFrames = len(step.frames)
    #
    for iFrame_step in range(0, numFrames):
        MODELframe = odb.steps[stepKey].frames[iFrame_step]
        #
        # Variable: U
        U_Field = MODELframe.fieldOutputs['U']
        U_RP_Zplus_SubField = U_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
        U_RP_Zminus_SubField = U_Field.getSubset(region=RP_Zminus_nSet, position=NODAL)
        #
        if isinstance(U_RP_Zplus_SubField.values[0].data, float):
            # Then variable is a scalar:
            U_RP_Zplus[iFrame_step] = U_RP_Zplus_SubField.values[0].data
            U_RP_Zminus[iFrame_step] = U_RP_Zminus_SubField.values[0].data
            #
        else:
            # Variable is an array:
            for j in range(0, len(U_RP_Zplus_SubField.values[0].data)):
                U_RP_Zplus[iFrame_step, j] = U_RP_Zplus_SubField.values[0].data[j]
                U_RP_Zminus[iFrame_step, j] = U_RP_Zminus_SubField.values[0].data[j]
                #
        #
        # Finished saving this variable!
        # Variable: UR
        UR_Field = MODELframe.fieldOutputs['UR']
        UR_RP_Zplus_SubField = UR_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
        UR_RP_Zminus_SubField = UR_Field.getSubset(region=RP_Zminus_nSet, position=NODAL)
        #
        if isinstance(UR_RP_Zplus_SubField.values[0].data, float):
            # Then variable is a scalar:
            UR_RP_Zplus[iFrame_step] = UR_RP_Zplus_SubField.values[0].data
            UR_RP_Zminus[iFrame_step] = UR_RP_Zminus_SubField.values[0].data
            #
        else:
            # Variable is an array:
            for j in range(0, len(UR_RP_Zplus_SubField.values[0].data)):
                UR_RP_Zplus[iFrame_step, j] = UR_RP_Zplus_SubField.values[0].data[j]
                UR_RP_Zminus[iFrame_step, j] = UR_RP_Zminus_SubField.values[0].data[j]
                #
        #
        # Finished saving this variable!
        # Variable: RF
        RF_Field = MODELframe.fieldOutputs['RF']
        RF_RP_Zplus_SubField = RF_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
        RF_RP_Zminus_SubField = RF_Field.getSubset(region=RP_Zminus_nSet, position=NODAL)
        #
        if isinstance(RF_RP_Zplus_SubField.values[0].data, float):
            # Then variable is a scalar:
            RF_RP_Zplus[iFrame_step] = RF_RP_Zplus_SubField.values[0].data
            RF_RP_Zminus[iFrame_step] = RF_RP_Zminus_SubField.values[0].data
            #
        else:
            # Variable is an array:
            for j in range(0, len(RF_RP_Zplus_SubField.values[0].data)):
                RF_RP_Zplus[iFrame_step, j] = RF_RP_Zplus_SubField.values[0].data[j]
                RF_RP_Zminus[iFrame_step, j] = RF_RP_Zminus_SubField.values[0].data[j]
                #
        #
        # Finished saving this variable!
        # Variable: RM
        RM_Field = MODELframe.fieldOutputs['RM']
        RM_RP_Zplus_SubField = RM_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
        RM_RP_Zminus_SubField = RM_Field.getSubset(region=RP_Zminus_nSet, position=NODAL)
        #
        if isinstance(RM_RP_Zplus_SubField.values[0].data, float):
            # Then variable is a scalar:
            RM_RP_Zplus[iFrame_step] = RM_RP_Zplus_SubField.values[0].data
            RM_RP_Zminus[iFrame_step] = RM_RP_Zminus_SubField.values[0].data
            #
        else:
            # Variable is an array:
            for j in range(0, len(RM_RP_Zplus_SubField.values[0].data)):
                RM_RP_Zplus[iFrame_step, j] = RM_RP_Zplus_SubField.values[0].data[j]
                RM_RP_Zminus[iFrame_step, j] = RM_RP_Zminus_SubField.values[0].data[j]

    STRUCTURE_variables = {'U': U_RP_Zplus, 'UR': UR_RP_Zplus,
                           'RF': RF_RP_Zplus, 'RM': RM_RP_Zplus}

    return STRUCTURE_variables


def _meshing(model, VertexPolygon, MastDiameter, MastPitch,
             cross_section_props, Emodulus, Gmodulus,
             ConeSlope, nStories=1, power=1., Twist_angle=0.,
             transition_length_ratio=1., include_name='include_mesh'):
    '''
    Parameters
    ----------
    VertexPolygon : int
        Number of vertices (sides) of the polygon base.
    MastDiameter : float
        Radius of the circumscribing circle of the polygon.
    MastPitch : float
        Pitch length of the strut (i.e. a single AstroMast!).
    cross_section_props : dict
        Stores the information about the cross-section. Specify the type
        of the cross section using 'type'. An empty 'type' will be
        understood as generalized cross section. Different types of
        sections are allowed:
            -'circular': requires 'd'
            -'generalized': requires 'Ixx', 'Iyy', 'J', 'area'
    Emodulus : float
        Youngus Modulus.
    Gmodulus : float
        Shear Modulus.
    ConeSlope : float
        Slope of the longerons (0 = straight, <0 larger at the top, >0 larger
        at the bottom).
    nStories : int
        Number of stories in HALF of the strut (i.e. in a single AstroMast!).
    power: float
        Power law exponent establishing the evolution of the spacing between
        battens.
    Twist_angle : float
        Do you want to twist the longerons?
    transition_length_ratio : float
        Transition zone for the longerons.
    '''

    # initialization
    MastRadius = MastDiameter / 2.0
    MastHeight = nStories * MastPitch
    # TODO: change section assignment
    Longeron_CS = cross_section_props['area']
    Ix = cross_section_props['Ixx']
    Iy = cross_section_props['Iyy']
    J = cross_section_props['J']

    Mesh_size = min(MastRadius, MastPitch) / 300.0

    # Create all the joints of the a single Deployable Mast:
    joints = np.zeros((nStories + 1, VertexPolygon, 3))
    joints_outter = np.zeros((nStories + 1, VertexPolygon, 3))
    for iStorey in range(0, nStories + 1, 1):
        for iVertex in range(0, VertexPolygon, 1):
            # Constant spacing between each storey (linear evolution):
            Zcoord = MastHeight / nStories * iStorey
            # Power-law spacing between each storey (more frequent at the fixed end):
            #        Zcoord = MastHeight*(float(iStorey)/float(nStories))**power
            # Power-law spacing between each storey (more frequent at the rotating end):
            #        Zcoord = -MastHeight/(float(nStories)**power)*(float(nStories-iStorey)**power)+MastHeight
            # Exponential spacing between each storey
            #        Zcoord =(MastHeight+1.0)/exp(float(nStories))*exp(float(iStorey))
            #
            Xcoord = MastRadius * np.cos(2.0 * np.pi / VertexPolygon * iVertex + Twist_angle * min(Zcoord / MastHeight / transition_length_ratio, 1.0))
            Ycoord = MastRadius * np.sin(2.0 * np.pi / VertexPolygon * iVertex + Twist_angle * min(Zcoord / MastHeight / transition_length_ratio, 1.0))
            # Save point defining this joint:
            joints[iStorey, iVertex, :] = (Xcoord * (1.0 - min(Zcoord, transition_length_ratio * MastHeight) / MastHeight * ConeSlope), Ycoord * (1.0 - min(Zcoord, transition_length_ratio * MastHeight) / MastHeight * ConeSlope), Zcoord)
            #
            # center = (0.0, 0.0)
            # vec = joints[iStorey, iVertex, 0:2] - center
            # norm_vec = np.linalg.norm(vec)
            joints_outter[iStorey, iVertex, 2] = joints[iStorey, iVertex, 2]
            joints_outter[iStorey, iVertex, 0:2] = joints[iStorey, iVertex, 0:2]
        # end iSide loop

    # end iStorey loop

    # Create the longerons:

    p_longerons = model.Part(name='longerons', dimensionality=THREE_D,
                             type=DEFORMABLE_BODY)

    p_longerons = model.parts['longerons']

    # d_longerons, r_longerons = p_longerons.datums, p_longerons.referencePoints

    LocalDatum_list = []  # List with local coordinate system for each longeron
    long_midpoints = []  # List with midpoints of longerons (just to determine a set containing the longerons)
    e_long = p_longerons.edges

    for iVertex in range(0, VertexPolygon, 1):
        # First create local coordinate system (useful for future constraints, etc.):
        iStorey = 0
        origin = joints[iStorey, iVertex, :]
        point2 = joints[iStorey, iVertex - 1, :]
        name = 'Local_Datum_' + str(iVertex)
        LocalDatum_list.append(p_longerons.DatumCsysByThreePoints(origin=origin, point2=point2, name=name,
                                                                  coordSysType=CARTESIAN, point1=(0.0, 0.0, 0.0)))
        #
        # Then, create the longerons
        templist = []  # List that will contain the points used to make each longeron
        for iStorey in range(0, nStories + 1, 1):
            templist.append(joints[iStorey, iVertex, :])
            if iStorey != 0:  # Save midpoints of bars
                long_midpoints.append([(joints[iStorey - 1, iVertex, :] + joints[iStorey, iVertex, :]) / 2, ])
            # end if
        # end iStorey loop
        p_longerons.WirePolyLine(points=templist,
                                 mergeType=IMPRINT, meshable=ON)
        # Create set for each longeron (to assign local beam directions)
        for i in range(0, len(templist)):  # loop over longerons edges
            if i == 0:
                select_edges = e_long.findAt([templist[0], ])  # Find the first edge
            else:
                # Now find remaining edges in longerons
                temp = e_long.findAt([templist[i], ])
                select_edges = select_edges + temp
            # end if
        # end i loop
        longeron_name = 'longeron-' + str(iVertex) + '_set'
        p_longerons.Set(edges=select_edges, name=longeron_name)

    # end for iVertex loop

    # Longerons set:
    e_long = p_longerons.edges
    select_edges = []
    for i in range(0, len(long_midpoints)):  # loop over longerons edges
        if i == 0:
            select_edges = e_long.findAt(long_midpoints[0])  # Find the first edge
        else:
            # Now find remaining edges in longerons
            temp = e_long.findAt(long_midpoints[i])
            select_edges = select_edges + temp
        # end if

    # end i loop

    p_longerons.Set(edges=select_edges, name='all_longerons_set')
    all_longerons_set_edges = select_edges

    p_longerons.Surface(circumEdges=all_longerons_set_edges, name='all_longerons_surface')

    # Create a set with all the joints:
    v_long = p_longerons.vertices
    select_vertices = []
    select_top_vertices = []
    select_bot_vertices = []
    for iStorey in range(0, nStories + 1, 1):
        for iVertex in range(0, VertexPolygon, 1):
            # Select all the joints in the longerons:
            current_joint = v_long.findAt([joints[iStorey, iVertex, :], ])  # Find the first vertex
            current_joint_name = 'joint-' + str(iStorey) + '-' + str(iVertex)
            # Create a set for each joint:
            p_longerons.Set(vertices=current_joint, name=current_joint_name)
            #
            if iStorey == 0 and iVertex == 0:
                select_vertices = current_joint  # Instantiate the first point in set
            else:
                select_vertices = select_vertices + current_joint  # Instantiate the first point in set
            # endif iStorey == 0 and iVertex == 0
            #
            if iStorey == 0:  # Also save the bottom nodes separately
                if iVertex == 0:
                    # Start selecting the bottom joints for implementing the boundary conditions
                    select_bot_vertices = current_joint
                else:
                    select_bot_vertices = select_bot_vertices + current_joint
                # endif iStorey == 0:
            elif iStorey == nStories:  # Also save the top nodes separately
                if iVertex == 0:
                    # Start selecting the top joints for implementing the boundary conditions
                    select_top_vertices = current_joint
                else:  # remaining vertices:
                    select_top_vertices = select_top_vertices + current_joint
            # end if
        # end iVertex loop

    # end iStorey loop

    p_longerons.Set(vertices=select_vertices, name='all_joints_set')
    p_longerons.Set(vertices=select_bot_vertices, name='bot_joints_set')
    p_longerons.Set(vertices=select_top_vertices, name='top_joints_set')

    #
    # Create materials:
    # model.Material(name='NiTi_alloy')
    # model.materials['NiTi_alloy'].Elastic(table=((83.0E3, 0.31),
    #                                              ))
    # model.materials['NiTi_alloy'].Density(table=((1.0E-3, ), ))

    # model.Material(name='PC')
    # model.materials['PC'].Elastic(table=((2134, 0.27),
    #                                      ))
    # model.materials['PC'].Density(table=((1.19E-3, ), ))

    # model.Material(name='PLA')
    # model.materials['PLA'].Elastic(table=((Emodulus, nu),
    #                                       ))
    # model.materials['PLA'].Density(table=((1.24E-3, ), ))

    # model.Material(name='CNT')
    # model.materials['CNT'].Elastic(table=((1000.0E3, 0.3),
    #                                       ))
    # model.materials['CNT'].Density(table=((1.0E-3, ), ))

    # Create beam profiles and beam sections:
    model.GeneralizedProfile(name='LongeronsProfile', area=Longeron_CS, i11=Ix, i12=0.0, i22=Iy, j=J, gammaO=0.0, gammaW=0.0)

    model.BeamSection(name='LongeronsSection', integration=BEFORE_ANALYSIS, poissonRatio=0.31, beamShape=CONSTANT,
                      profile='LongeronsProfile', density=0.00124, thermalExpansion=OFF,
                      temperatureDependency=OFF, dependencies=0, table=((Emodulus, Gmodulus), ),
                      alphaDamping=0.0, betaDamping=0.0, compositeDamping=0.0, centroid=(0.0,
                                                                                         0.0), shearCenter=(0.0, 0.0), consistentMassMatrix=False)

    # Assign respective sections:
    p_longerons.SectionAssignment(offset=0.0,
                                  offsetField='', offsetType=MIDDLE_SURFACE, region=p_longerons.sets['all_longerons_set'],
                                  sectionName='LongeronsSection', thicknessAssignment=FROM_SECTION)

    # Assing beam orientation:
    for iVertex in range(0, VertexPolygon, 1):
        iStorey = 0
        dir_vec_n1 = joints[iStorey, iVertex, :] - (0., 0., 0.)  # Vector n1 perpendicular to the longeron tangent
        longeron_name = 'longeron-' + str(iVertex) + '_set'
        region = p_longerons.sets[longeron_name]
        p_longerons.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=dir_vec_n1)

    # end for iVertex
    #

    # delta = Mesh_size / 100.0
    ########################################################################
    # Mesh the structure

    # refPlane = p_longerons.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=L/2)
    # d = p.datums
    # All_faces = facesLeafs+facesDoubleThickBoom
    # p.PartitionFaceByDatumPlane(datumPlane=d[refPlane.id], faces=All_faces)
    # #
    # p = model.parts['reducedCF_TRAC_boom']

    p_longerons.seedPart(size=Mesh_size, deviationFactor=0.04, minSizeFactor=0.001,
                         constraint=FINER)
    p_longerons.seedEdgeBySize(edges=all_longerons_set_edges, size=Mesh_size, deviationFactor=0.04,
                               constraint=FINER)
    elemType_longerons = mesh.ElemType(elemCode=B31, elemLibrary=STANDARD)  # Element type
    p_longerons.setElementType(regions=(all_longerons_set_edges, ), elemTypes=(elemType_longerons, ))
    p_longerons.generateMesh()

    #######################################################################

    # Make Analytical surfaces for contact purposes
    s1 = model.ConstrainedSketch(name='__profile__',
                                 sheetSize=MastRadius * 3.0)
    # g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    g = s1.geometry
    s1.setPrimaryObject(option=STANDALONE)
    s1.Line(point1=(0.0, -MastRadius * 1.1), point2=(0.0, MastRadius * 1.1))
    s1.VerticalConstraint(entity=g[2], addUndoState=False)
    p_surf = model.Part(name='AnalyticSurf', dimensionality=THREE_D,
                        type=ANALYTIC_RIGID_SURFACE)
    p_surf = model.parts['AnalyticSurf']
    p_surf.AnalyticRigidSurfExtrude(sketch=s1, depth=MastRadius * 2.2)
    s1.unsetPrimaryObject()

    rigid_face = p_surf.faces
    # surf_select = f.findAt((0.0,MastRadius*1.05,0.0))
    # surf_select = f[0]
    p_surf.Surface(side1Faces=rigid_face, name='rigid_support')
    # p_surf.Set(faces=surf_select, name='support_surface_set')
    # p_surf.sets['all_diagonals_set']

    #
    # Make assembly:
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    # Create reference points to assign boundary conditions
    RP_ZmYmXm = a.ReferencePoint(point=(0.0, 0.0, -1.1 * MastRadius))
    refpoint_ZmYmXm = (a.referencePoints[RP_ZmYmXm.id],)
    a.Set(referencePoints=refpoint_ZmYmXm, name='RP_ZmYmXm')
    #
    RP_ZpYmXm = a.ReferencePoint(point=(0.0, 0.0, MastHeight + 1.1 * MastRadius))
    refpoint_ZpYmXm = (a.referencePoints[RP_ZpYmXm.id],)
    a.Set(referencePoints=refpoint_ZpYmXm, name='RP_ZpYmXm')
    #
    # Create longerons
    a_long = a.Instance(name='longerons-1-1', part=p_longerons, dependent=ON)
    # Create bottom surface
    a_surf_bot = a.Instance(name='AnalyticSurf-1-1', part=p_surf, dependent=ON)
    # Now rotate the plane to have the proper direction
    a.rotate(instanceList=('AnalyticSurf-1-1', ), axisPoint=(0.0, 0.0, 0.0),
             axisDirection=(0.0, 1.0, 0.0), angle=90.0)
    #
    # Create set with surface
    select_bot_surf = a_surf_bot.surfaces['rigid_support']
    # Perhaps we need to define a set instead of a face
    # AnalyticSurf_surface=a_surf_bot.Surface(side1Faces=select_bot_surf, name='support_surf_bot-1')
    model.RigidBody(name='Constraint-RigidBody_surf_bot-1', refPointRegion=refpoint_ZmYmXm,
                    surfaceRegion=select_bot_surf)
    for iVertex in range(0, VertexPolygon, 1):
        #
        # Select appropriate coordinate system:
        DatumID = LocalDatum_list[iVertex].id
        datum = a_long.datums[DatumID]
        for iStorey in range(0, nStories + 1, 1):
            # Current joint:
            current_joint_name = 'joint-' + str(iStorey) + '-' + str(iVertex)
            # Define COUPLING constraints for all the joints:
            if iStorey == 0:  # Bottom base:
                #
                master_region = a.sets['RP_ZmYmXm']  # Note that the master is the Reference Point
                #
                slave_region = a_long.sets[current_joint_name]
                # Make constraint for this joint:
                Constraint_name = 'RP_ZmYmXm_PinConstraint-' + str(iStorey) + '-' + str(iVertex)
                model.Coupling(name=Constraint_name, controlPoint=master_region,
                               surface=slave_region, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC,
                               localCsys=datum, u1=ON, u2=ON, u3=ON, ur1=OFF, ur2=ON, ur3=ON)
                #
                # Constraint_name = 'RP_ZmYmXm_FixedConstraint-'+str(iStorey)+'-'+str(iVertex)
                # model.Coupling(name=Constraint_name, controlPoint=master_region,
                #    surface=slave_region, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC,
                #    localCsys=datum, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
                # Make constraint for this joint:
            elif iStorey == nStories:  # Top base:
                #
                master_region = a.sets['RP_ZpYmXm']  # Note that the master is the Reference Point
                #
                slave_region = a_long.sets[current_joint_name]
                # Make constraint for this joint:
                Constraint_name = 'RP_ZpYmXm_PinConstraint-' + str(iStorey) + '-' + str(iVertex)
                model.Coupling(name=Constraint_name, controlPoint=master_region,
                               surface=slave_region, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC,
                               localCsys=datum, u1=ON, u2=ON, u3=ON, ur1=OFF, ur2=ON, ur3=ON)
                #
                # Constraint_name = 'RP_ZpYmXm_FixedConstraint-'+str(iStorey)+'-'+str(iVertex)
                # model.Coupling(name=Constraint_name, controlPoint=master_region,
                #    surface=slave_region, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC,
                #    localCsys=datum, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
                # Make constraint for this joint:
            else:  # Middle stories:
                master_region = a_long.sets[current_joint_name]
                #
                slave_region = a_bat.sets[current_joint_name]
                # Make constraint for this joint:
            # endif iStorey
            #
        # end for iStorey

    # end for iVertex

    #

    # Create hinges:
    # select_joints=a.instances['deployable_mast-1'].sets['all_joints_set']
    # select_RefPoint=a.sets['RP_joints']
    # model.RigidBody(name='JointsContraint', refPointRegion=select_RefPoint,
    #    pinRegion=select_joints)

    #
    # Export mesh to .inp file
    #
    modelJob = mdb.Job(name=include_name, model=model.name)
    modelJob.writeInput(consistencyChecking=OFF)
    # End of python script
