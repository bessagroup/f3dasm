#=====================================================================#
#
# Created by M.A. Bessa on 12-Nov-2019 07:06:42
#=====================================================================#
from abaqusConstants import *
from odbAccess import *
import os
import numpy
import collections
from copy import deepcopy
try:
    import cPickle as pickle  # Improve speed
except ValueError:
    import pickle

def dict_merge(a, b):
    #recursively merges dict's. not just simple a['key'] = b['key'], if
    #both a and b have a key who's value is a dict then dict_merge is called
    #on both values and the result stored in the returned dictionary.
    if not isinstance(b, dict):
        return b
    result = deepcopy(a)
    for k, v in b.iteritems():
        if k in result and isinstance(result[k], dict):
                result[k] = dict_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result

#
os.chdir(r'/home/gkus/F3DAS-master/3_Analyses/DOE_Ix-PD-100/Input_point1/Imperfection_point1/DoE_point100')
# Set directory where you want the post-processing file to be generated
postproc_dir='/home/gkus/F3DAS-master/4_Postprocessing/DOE_Ix-PD-100'
#
file_postproc_path = postproc_dir+'/'+'STRUCTURES_postprocessing_variables_CPU1.p'
# Flag saying post-processing file exists:
try:
    # Try to load a previous post-processing file with all information:
    readfile_postproc = open(file_postproc_path, 'rb')
    STRUCTURES_data_old = pickle.load(readfile_postproc)
    readfile_postproc.close()
    postproc_exists = 1
except Exception, e:
    postproc_exists = 0 # Flag saying that there is no post-processing file saved previously

#
# Name of the file job
jobname='DoE100_riks'
#
odbfile=jobname+'.odb' # Define name of this .odb file
try:
    # Try to open this odb file
    MODELodb = openOdb(path=odbfile) # Open .odb file
except Exception, e:
    print >> sys.stderr, 'does not exist'
    print >> sys.stderr, 'Exception: %s' % str(e)
    sys.exit(1) # Exit the code because there is nothing left to do!

#
# Determine the number of steps in the output database.
mySteps = MODELodb.steps
numSteps = len(mySteps)
#
RP_Zplus_nSet = MODELodb.rootAssembly.nodeSets['RP_ZPYMXM']
RP_Zminus_nSet = MODELodb.rootAssembly.nodeSets['RP_ZMYMXM']
entireSTRUCTURE_nSet = MODELodb.rootAssembly.nodeSets[' ALL NODES']
entireSTRUCTURE_elSet = MODELodb.rootAssembly.elementSets[' ALL ELEMENTS']
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
MODELframe = MODELodb.steps[mySteps.keys()[0]].frames[0] # Undeformed config.
U_Field = MODELframe.fieldOutputs['U']
U_RP_Zplus_SubField = U_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
U_RP_Zminus_SubField = U_Field.getSubset(region=RP_Zminus_nSet, position=NODAL)
#
if isinstance(U_RP_Zplus_SubField.values[0].data,float):
    # Then variable is a scalar
    U_RP_Zplus = numpy.zeros(( totalNumFrames ))
    U_RP_Zminus = numpy.zeros(( totalNumFrames ))
else:
    # Variable is an array
    U_RP_Zplus = numpy.zeros(( totalNumFrames,len(U_RP_Zplus_SubField.values[0].data) ))
    U_RP_Zminus = numpy.zeros(( totalNumFrames,len(U_RP_Zminus_SubField.values[0].data) ))

UR_Field = MODELframe.fieldOutputs['UR']
UR_RP_Zplus_SubField = UR_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
UR_RP_Zminus_SubField = UR_Field.getSubset(region=RP_Zminus_nSet, position=NODAL)
#
if isinstance(UR_RP_Zplus_SubField.values[0].data,float):
    # Then variable is a scalar
    UR_RP_Zplus = numpy.zeros(( totalNumFrames ))
    UR_RP_Zminus = numpy.zeros(( totalNumFrames ))
else:
    # Variable is an array
    UR_RP_Zplus = numpy.zeros(( totalNumFrames,len(UR_RP_Zplus_SubField.values[0].data) ))
    UR_RP_Zminus = numpy.zeros(( totalNumFrames,len(UR_RP_Zminus_SubField.values[0].data) ))

RF_Field = MODELframe.fieldOutputs['RF']
RF_RP_Zplus_SubField = RF_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
RF_RP_Zminus_SubField = RF_Field.getSubset(region=RP_Zminus_nSet, position=NODAL)
#
if isinstance(RF_RP_Zplus_SubField.values[0].data,float):
    # Then variable is a scalar
    RF_RP_Zplus = numpy.zeros(( totalNumFrames ))
    RF_RP_Zminus = numpy.zeros(( totalNumFrames ))
else:
    # Variable is an array
    RF_RP_Zplus = numpy.zeros(( totalNumFrames,len(RF_RP_Zplus_SubField.values[0].data) ))
    RF_RP_Zminus = numpy.zeros(( totalNumFrames,len(RF_RP_Zminus_SubField.values[0].data) ))

RM_Field = MODELframe.fieldOutputs['RM']
RM_RP_Zplus_SubField = RM_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
RM_RP_Zminus_SubField = RM_Field.getSubset(region=RP_Zminus_nSet, position=NODAL)
#
if isinstance(RM_RP_Zplus_SubField.values[0].data,float):
    # Then variable is a scalar
    RM_RP_Zplus = numpy.zeros(( totalNumFrames ))
    RM_RP_Zminus = numpy.zeros(( totalNumFrames ))
else:
    # Variable is an array
    RM_RP_Zplus = numpy.zeros(( totalNumFrames,len(RM_RP_Zplus_SubField.values[0].data) ))
    RM_RP_Zminus = numpy.zeros(( totalNumFrames,len(RM_RP_Zminus_SubField.values[0].data) ))

# Loop over the Frames of this Step to extract values of variables
stepTotalTime = numpy.zeros(numSteps)
previousFrame = 0
numFrames = 0
stepKey = mySteps.keys()[0]
step = mySteps[stepKey]
stepTotalTime = step.timePeriod
numFrames = len(step.frames)
#
for iFrame_step in range (0,numFrames):
    MODELframe = MODELodb.steps[stepKey].frames[iFrame_step]
    #
    # Variable: U
    U_Field = MODELframe.fieldOutputs['U']
    U_RP_Zplus_SubField = U_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
    U_RP_Zminus_SubField = U_Field.getSubset(region=RP_Zminus_nSet, position=NODAL)
    #
    if isinstance(U_RP_Zplus_SubField.values[0].data,float):
        # Then variable is a scalar:
        U_RP_Zplus[iFrame_step] = U_RP_Zplus_SubField.values[0].data
        U_RP_Zminus[iFrame_step] = U_RP_Zminus_SubField.values[0].data
        #
    else:
        # Variable is an array:
        for j in range(0, len(U_RP_Zplus_SubField.values[0].data)):
            U_RP_Zplus[iFrame_step,j] = U_RP_Zplus_SubField.values[0].data[j]
            U_RP_Zminus[iFrame_step,j] = U_RP_Zminus_SubField.values[0].data[j]
            #
    #
    # Finished saving this variable!
    # Variable: UR
    UR_Field = MODELframe.fieldOutputs['UR']
    UR_RP_Zplus_SubField = UR_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
    UR_RP_Zminus_SubField = UR_Field.getSubset(region=RP_Zminus_nSet, position=NODAL)
    #
    if isinstance(UR_RP_Zplus_SubField.values[0].data,float):
        # Then variable is a scalar:
        UR_RP_Zplus[iFrame_step] = UR_RP_Zplus_SubField.values[0].data
        UR_RP_Zminus[iFrame_step] = UR_RP_Zminus_SubField.values[0].data
        #
    else:
        # Variable is an array:
        for j in range(0, len(UR_RP_Zplus_SubField.values[0].data)):
            UR_RP_Zplus[iFrame_step,j] = UR_RP_Zplus_SubField.values[0].data[j]
            UR_RP_Zminus[iFrame_step,j] = UR_RP_Zminus_SubField.values[0].data[j]
            #
    #
    # Finished saving this variable!
    # Variable: RF
    RF_Field = MODELframe.fieldOutputs['RF']
    RF_RP_Zplus_SubField = RF_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
    RF_RP_Zminus_SubField = RF_Field.getSubset(region=RP_Zminus_nSet, position=NODAL)
    #
    if isinstance(RF_RP_Zplus_SubField.values[0].data,float):
        # Then variable is a scalar:
        RF_RP_Zplus[iFrame_step] = RF_RP_Zplus_SubField.values[0].data
        RF_RP_Zminus[iFrame_step] = RF_RP_Zminus_SubField.values[0].data
        #
    else:
        # Variable is an array:
        for j in range(0, len(RF_RP_Zplus_SubField.values[0].data)):
            RF_RP_Zplus[iFrame_step,j] = RF_RP_Zplus_SubField.values[0].data[j]
            RF_RP_Zminus[iFrame_step,j] = RF_RP_Zminus_SubField.values[0].data[j]
            #
    #
    # Finished saving this variable!
    # Variable: RM
    RM_Field = MODELframe.fieldOutputs['RM']
    RM_RP_Zplus_SubField = RM_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
    RM_RP_Zminus_SubField = RM_Field.getSubset(region=RP_Zminus_nSet, position=NODAL)
    #
    if isinstance(RM_RP_Zplus_SubField.values[0].data,float):
        # Then variable is a scalar:
        RM_RP_Zplus[iFrame_step] = RM_RP_Zplus_SubField.values[0].data
        RM_RP_Zminus[iFrame_step] = RM_RP_Zminus_SubField.values[0].data
        #
    else:
        # Variable is an array:
        for j in range(0, len(RM_RP_Zplus_SubField.values[0].data)):
            RM_RP_Zplus[iFrame_step,j] = RM_RP_Zplus_SubField.values[0].data[j]
            RM_RP_Zminus[iFrame_step,j] = RM_RP_Zminus_SubField.values[0].data[j]
            #
    #
    # Finished saving this variable!
    # Now compute other local quantities:

#
# Save variables of interest to a database
STRUCTURE_variables = {'riks_RP_Zplus_U':U_RP_Zplus,'riks_RP_Zplus_UR':UR_RP_Zplus,'riks_RP_Zplus_RF':RF_RP_Zplus,'riks_RP_Zplus_RM':RM_RP_Zplus}
#
stringDoE = 'DoE'+str(100)
stringImperfection = 'Imperfection'+str(1)
stringInput = 'Input'+str(1)
STRUCTURES_data_update = {stringInput : {stringImperfection : {stringDoE : STRUCTURE_variables } } }
if postproc_exists == 1:
    STRUCTURES_data = dict_merge(STRUCTURES_data_old, STRUCTURES_data_update)
else:
    STRUCTURES_data = STRUCTURES_data_update

# Save post-processing information to pickle file:
writefile_postproc = open(file_postproc_path, 'wb')
pickle.dump(STRUCTURES_data, writefile_postproc)
writefile_postproc.close()

#
# End of file.
