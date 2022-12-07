# import packages for abaqus post-processing
from odbAccess import *
import os
import numpy

try:
    import cPickle as pickle  # Improve speed
except ValueError:
    import pickle


def main(sim_info):
    """

    Parameters
    ----------
    Job_name: str of the

    Returns
    -------

    """
    # basic information of the odb file
    # jobname='Job-1'
    # Define the name of the Phase_1 part
    job_name = str(sim_info["job_name"])
    odbfile = job_name + ".odb"  # Define name of this .odb file
    RVEodb = openOdb(path=odbfile)

    # Determine the number of steps in the output database.
    mySteps = RVEodb.steps
    numSteps = len(mySteps)
    #
    # Phase_1_nSet = RVEodb.rootAssembly.instances['FINALRVE'].nodeSets['PHASE_1']
    # Phase_1_elSet = RVEodb.rootAssembly.instances['FINALRVE'].elementSets['PHASE_1']
    # botEdge_nSet = RVEodb.rootAssembly.instances['FINALRVE'].nodeSets['BOTEDGE']
    # botEdge_elSet = RVEodb.rootAssembly.instances['FINALRVE'].elementSets['BOTEDGE']
    # rightEdge_nSet = RVEodb.rootAssembly.instances['FINALRVE'].nodeSets['RIGHTEDGE']
    # rightEdge_elSet = RVEodb.rootAssembly.instances['FINALRVE'].elementSets['RIGHTEDGE']
    # topEdge_nSet = RVEodb.rootAssembly.instances['FINALRVE'].nodeSets['TOPEDGE']
    # topEdge_elSet = RVEodb.rootAssembly.instances['FINALRVE'].elementSets['TOPEDGE']
    # leftEdge_nSet = RVEodb.rootAssembly.instances['FINALRVE'].nodeSets['LEFTEDGE']
    # leftEdge_elSet = RVEodb.rootAssembly.instances['FINALRVE'].elementSets['LEFTEDGE']
    # entireRVE_nSet = RVEodb.rootAssembly.nodeSets[' ALL NODES']
    entireRVE_elSet = RVEodb.rootAssembly.elementSets[" ALL ELEMENTS"]
    dummy1_nSet = RVEodb.rootAssembly.nodeSets["REF-R"]
    dummy2_nSet = RVEodb.rootAssembly.nodeSets["REF-T"]
    Lx = 3.50000e00  # RVE dimension along x
    Ly = 3.50000e00  # RVE dimension along y
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

    # Preallocate quantities for speed
    RVEframe = RVEodb.steps[mySteps.keys()[0]].frames[0]  # Undeformed config.
    # Extract volume at integration point in ENTIRE RVE:
    ivolField = RVEframe.fieldOutputs["IVOL"]
    ivolSubField = ivolField.getSubset(
        region=entireRVE_elSet, position=INTEGRATION_POINT)
    #
    ivol = numpy.zeros((len(ivolSubField.values)))
    tot_vol = 0.0
    for i in range(0, len(ivolSubField.values)):
        # Volume for i-th integration point
        ivol[i] = ivolSubField.values[i].data
        tot_vol = tot_vol + ivol[i]  # total volume

    # finished computing volume at integration points and total volume
    ELSE_Field = RVEframe.fieldOutputs["ELSE"]
    ELSE_SubField = ELSE_Field.getSubset(
        region=entireRVE_elSet, position=WHOLE_ELEMENT)
    #
    if isinstance(ELSE_SubField.values[0].data, float):
        # Then variable is a scalar
        av_ELSE = numpy.zeros((totalNumFrames))
    else:
        # Variable is an array
        av_ELSE = numpy.zeros(
            (totalNumFrames, len(ELSE_SubField.values[0].data)))

    U_Field = RVEframe.fieldOutputs["U"]
    U_dummy1_SubField = U_Field.getSubset(region=dummy1_nSet, position=NODAL)
    U_dummy2_SubField = U_Field.getSubset(region=dummy2_nSet, position=NODAL)
    #
    if isinstance(U_dummy1_SubField.values[0].data, float):
        # Then variable is a scalar
        U_dummy1 = numpy.zeros((totalNumFrames))
        U_dummy2 = numpy.zeros((totalNumFrames))
    else:
        # Variable is an array
        U_dummy1 = numpy.zeros(
            (totalNumFrames, len(U_dummy1_SubField.values[0].data)))
        U_dummy2 = numpy.zeros(
            (totalNumFrames, len(U_dummy2_SubField.values[0].data)))

    RF_Field = RVEframe.fieldOutputs["RF"]
    RF_dummy1_SubField = RF_Field.getSubset(region=dummy1_nSet, position=NODAL)
    RF_dummy2_SubField = RF_Field.getSubset(region=dummy2_nSet, position=NODAL)
    #
    if isinstance(RF_dummy1_SubField.values[0].data, float):
        # Then variable is a scalar
        RF_dummy1 = numpy.zeros((totalNumFrames))
        RF_dummy2 = numpy.zeros((totalNumFrames))
    else:
        # Variable is an array
        RF_dummy1 = numpy.zeros(
            (totalNumFrames, len(RF_dummy1_SubField.values[0].data)))
        RF_dummy2 = numpy.zeros(
            (totalNumFrames, len(RF_dummy2_SubField.values[0].data)))

    # Loop over Steps and Frames to compute average quantities in RVE
    eye = numpy.identity(2)
    defGrad = numpy.zeros((totalNumFrames, 2, 2))
    nomP = numpy.zeros((totalNumFrames, 2, 2))
    jacobian = numpy.zeros(totalNumFrames)
    Green_strain = numpy.zeros((totalNumFrames, 2, 2))
    PK2 = numpy.zeros((totalNumFrames, 2, 2))
    stepTotalTime = numpy.zeros(numSteps)
    previousFrame = 0
    numFrames = 0
    for iStep in range(numSteps):
        previousFrame = previousFrame + numFrames
        stepKey = mySteps.keys()[iStep]
        step = mySteps[stepKey]
        stepTotalTime[iStep] = step.timePeriod
        numFrames = len(step.frames)
        #
        for iFrame_step in range(0, numFrames):
            iFrame = previousFrame + iFrame_step
            RVEframe = RVEodb.steps[stepKey].frames[iFrame]
            #
            # Variable: ELSE
            ELSE_Field = RVEframe.fieldOutputs["ELSE"]
            ELSE_SubField = ELSE_Field.getSubset(
                region=entireRVE_elSet, position=WHOLE_ELEMENT)
            #
            if isinstance(ELSE_SubField.values[0].data, float):
                # Then variable is a scalar:
                # Loop over every element to compute average
                for i in range(0, len(ELSE_SubField.values)):
                    av_ELSE[iFrame] = av_ELSE[iFrame] + \
                        ELSE_SubField.values[i].data
                #
                av_ELSE[iFrame] = av_ELSE[iFrame] / tot_vol
            else:
                # Variable is an array:
                # Loop over every element to compute average
                for j in range(0, len(ELSE_SubField.values[0].data)):
                    for i in range(0, len(ELSE_SubField.values)):
                        av_ELSE[iFrame][j] = av_ELSE[iFrame][j] + \
                            ELSE_SubField.values[i].data[j]
                    #
                    av_ELSE[iFrame][j] = av_ELSE[iFrame][j] / tot_vol
                    av_ELSE[iFrame][j] = av_  #
            # Finished computing average for this variable!
            # Variable: U
            U_Field = RVEframe.fieldOutputs["U"]
            U_dummy1_SubField = U_Field.getSubset(
                region=dummy1_nSet, position=NODAL)
            U_dummy2_SubField = U_Field.getSubset(
                region=dummy2_nSet, position=NODAL)
            #
            if isinstance(U_dummy1_SubField.values[0].data, float):
                # Then variable is a scalar:
                U_dummy1[iFrame] = U_dummy1_SubField.values[0].data
                U_dummy2[iFrame] = U_dummy2_SubField.values[0].data
                #
            else:
                # Variable is an array:
                for j in range(0, len(U_dummy1_SubField.values[0].data)):
                    U_dummy1[iFrame][j] = U_dummy1_SubField.values[0].data[j]
                    U_dummy2[iFrame][j] = U_dummy2_SubField.values[0].data[j]

            # Finished saving this variable at the dummy nodes!
            # Variable: RF
            RF_Field = RVEframe.fieldOutputs["RF"]
            RF_dummy1_SubField = RF_Field.getSubset(
                region=dummy1_nSet, position=NODAL)
            RF_dummy2_SubField = RF_Field.getSubset(
                region=dummy2_nSet, position=NODAL)
            #
            if isinstance(RF_dummy1_SubField.values[0].data, float):
                # Then variable is a scalar:
                RF_dummy1[iFrame] = RF_dummy1_SubField.values[0].data
                RF_dummy2[iFrame] = RF_dummy2_SubField.values[0].data
                #
            else:
                # Variable is an array:
                for j in range(0, len(RF_dummy1_SubField.values[0].data)):
                    RF_dummy1[iFrame][j] = RF_dummy1_SubField.values[0].data[j]
                    RF_dummy2[iFrame][j] = RF_dummy2_SubField.values[0].data[j]

            # Finished saving this variable at the dummy nodes!
            # Now compute the deformation gradient, jacobian and nominal stress:
            for j in range(0, 2):
                defGrad[iFrame][0][j] = U_dummy1[iFrame][j] + eye[0][j]
                defGrad[iFrame][1][j] = U_dummy2[iFrame][j] + eye[1][j]
                nomP[iFrame][0][j] = RF_dummy1[iFrame][j] / tot_vol
                nomP[iFrame][1][j] = RF_dummy2[iFrame][j] / tot_vol
            jacobian[iFrame] = numpy.linalg.det(defGrad[iFrame][:][:])
            Green_strain[iFrame, :, :] = 0.5 * (
                numpy.dot(numpy.transpose(
                    defGrad[iFrame, :, :]), defGrad[iFrame, :, :]) - numpy.identity(2)
            )
            PK2[iFrame, :, :] = numpy.dot(nomP[iFrame, :, :], numpy.linalg.inv(
                numpy.transpose(defGrad[iFrame, :, :])))

    # Save all variables to a single structured variable with all the data
    RVE_variables = {
        "ELSE": av_ELSE,
        "step_total_time": stepTotalTime,
        "P": nomP,
        "F": defGrad,
        "J": jacobian,
        "total_vol": tot_vol,
        "Green_strain": Green_strain,
        "PK2": PK2,
    }
    # Save post-processing information to pkl file:
    with open("results.p", "w") as fp:
        pickle.dump(RVE_variables, fp)
