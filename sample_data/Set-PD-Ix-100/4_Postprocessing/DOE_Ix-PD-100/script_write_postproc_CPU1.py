#=====================================================================#
#
# Created by M.A. Bessa on 12-Nov-2019 07:05:11
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
import re
match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?')
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
try:
    # Try to open this odb file
    #Import Abaqus odb for the Buckle analysis
    MODELodb=openOdb('DoE100_linear_buckle'+'.odb')
    #
except Exception, e:
    print >> sys.stderr, 'does not exist'
    print >> sys.stderr, 'Exception: %s' % str(e)
    sys.exit(1) # Exit the code because there is nothing left to do!
#

# Determine the number of steps in the output database.
mySteps = MODELodb.steps
numSteps = len(mySteps)
stepKey = mySteps.keys()[0]
step = mySteps[stepKey]
numFrames = len(step.frames)
#
maxDisp = numpy.zeros(( numFrames ))
RP_Zplus_nSet = MODELodb.rootAssembly.nodeSets['RP_ZPYMXM']
RP_Zminus_nSet = MODELodb.rootAssembly.nodeSets['RP_ZMYMXM']
entireSTRUCTURE_nSet = MODELodb.rootAssembly.nodeSets[' ALL NODES']
entireSTRUCTURE_elSet = MODELodb.rootAssembly.elementSets[' ALL ELEMENTS']
#
# Read critical buckling load
P_crit=numpy.zeros( numFrames-1 )
coilable=numpy.zeros( numFrames-1 )
for iFrame_step in range(1,numFrames):
    # Read critical buckling load
    MODELframe = MODELodb.steps[mySteps.keys()[0]].frames[iFrame_step]
    eigenValue = [float(x) for x in re.findall(match_number, MODELframe.description)]
    #eigenValue[1] is the eigenValue for Mode = eigenValue[0]
    P_crit[iFrame_step-1]=eigenValue[1]
    #Now check if this is a coilable mode
    UR_Field = MODELframe.fieldOutputs['UR']
    UR_SubField = UR_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
    UR_Mode1_RP_ZpYmXm = UR_SubField.values[0].data
    U_Field = MODELframe.fieldOutputs['U']
    U_SubField = U_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)
    U_Mode1_RP_ZpYmXm = U_SubField.values[0].data
    if (abs(UR_Mode1_RP_ZpYmXm[2]) > 1.0e-4) and (abs(U_Mode1_RP_ZpYmXm[0]) < 1.0e-4) and (abs(U_Mode1_RP_ZpYmXm[1]) < 1.0e-4):
        coilable[iFrame_step-1] = 1
    else:
        coilable[iFrame_step-1] = 0

#
UR_Field = MODELframe.fieldOutputs['UR']
UR_SubField = UR_Field.getSubset(region=entireSTRUCTURE_nSet, position=NODAL)
#
if isinstance(UR_SubField.values[0].data,float):
    # Then variable is a scalar
    max_UR = numpy.ones(( numFrames ))*(-1e20)
else:
    # Variable is an array
    max_UR = numpy.ones(( numFrames,len(UR_SubField.values[0].data) ))*(-1e20)

for iFrame_step in range(numFrames):
    # Read frame
    MODELframe = MODELodb.steps[mySteps.keys()[0]].frames[iFrame_step]
    # Variable: UR
    UR_Field = MODELframe.fieldOutputs['UR']
    UR_SubField = UR_Field.getSubset(region=entireSTRUCTURE_nSet, position=NODAL)
    #
    if isinstance(UR_SubField.values[0].data,float):
        # Then variable is a scalar:
        for strainValue in UR_SubField.values:
            if (strainValue.data > max_UR[iFrame_step]):
                max_UR[iFrame_step] = abs(strainValue.data[0])
            #
    else:
        # Variable is an array:
        for strainValue in UR_SubField.values:
            for j in range(0, len(UR_SubField.values[0].data)):
                if (strainValue.data[j] > max_UR[iFrame_step,j]):
                    max_UR[iFrame_step,j] = abs(strainValue.data[j])
                    max_UR[iFrame_step,j] = abs(strainValue.data[j])
                #
    #
    # Finished saving this variable!
    # Now compute the MAXIMUM of the 3 components of U for this frame:
    maxDisp[iFrame_step] = max(max_UR[iFrame_step,:]) 
#
# Save variables of interest to a database
STRUCTURE_variables = {'P_p3_crit':P_crit, 'maxDisp_p3':maxDisp, 'coilable':coilable}
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

# End of file.
