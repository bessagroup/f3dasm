#=====================================================================#
#
# Created by M.A. Bessa on 12-Nov-2019 05:43:30
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
os.chdir(r'/home/gkus/F3DAS-master/3_Analyses/DOE_Ix-PD-100/Input_point1/Imperfection_point1/DoE_point74')
# Set directory where the post-processing file is
postproc_dir='/home/gkus/F3DAS-master/4_Postprocessing/DOE_Ix-PD-100'
#
file_postproc_path = postproc_dir+'/'+'STRUCTURES_postprocessing_variables_CPU1.p'
# Flag saying post-processing file exists:
try:
    # Try to load a previous post-processing file with all information:
    readfile_postproc = open(file_postproc_path, 'rb')
    STRUCTURES_data = pickle.load(readfile_postproc)
    readfile_postproc.close()
    postproc_exists = 1
except Exception, e:
    postproc_exists = 0 # Flag saying that there is no post-processing file saved previously
    sys.exit(1) # Exit the code because there is nothing left to do!

#
with open('DoE74_riks.inp','wb') as File:
    if STRUCTURES_data['Input1']['Imperfection1']['DoE74']['coilable'][0] == 0:
        sys.exit() # do not bother running the RIKS analysis because the material will not coil...
    #
    File.write('** Include file with mesh of structure:\n')
    File.write('*INCLUDE, INPUT=include_mesh_DoE74.inp\n')
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
    File.write('*IMPERFECTION, FILE=DoE74_linear_buckle, STEP=1\n')
    mode_amplitude = 8.70312e-02/STRUCTURES_data['Input1']['Imperfection1']['DoE74']['maxDisp_p3'][1]
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
    File.write('RP_ZpYmXm, 3, 3, -4.76098e+01\n')
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

#
