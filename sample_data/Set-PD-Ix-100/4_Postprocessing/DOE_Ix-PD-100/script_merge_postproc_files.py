#=====================================================================#
#
# Created by M.A. Bessa on 12-Nov-2019 07:06:54
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
# Set directory where you want the post-processing file to be generated
postproc_dir='/home/gkus/F3DAS-master/4_Postprocessing/DOE_Ix-PD-100'
#
file_postproc_path = postproc_dir+'/'+'STRUCTURES_postprocessing_variables.p'
# Flag saying that the MERGED (unique) post-processing file exists:
try:
    # Try to load the previous (unique) post-processing file with all information:
    readfile_postproc = open(file_postproc_path, 'rb')
    STRUCTURES_data = pickle.load(readfile_postproc)
    readfile_postproc.close()
    postproc_exists = 1
except Exception, e:
    postproc_exists = 0 # Flag saying that there is no post-processing file saved previously

#
file_postproc_path_CPU1 = postproc_dir+'/'+'STRUCTURES_postprocessing_variables_CPU1.p'
# Flag saying post-processing file for this CPU exists:
try:
    # Try to load the previous post-processing file for this CPU with all information:
    readfile_postproc_CPU1 = open(file_postproc_path_CPU1, 'rb')
    STRUCTURES_data_CPU1 = pickle.load(readfile_postproc_CPU1)
    readfile_postproc_CPU1.close()
    postproc_exists_CPU1 = 1
except Exception, e:
    postproc_exists_CPU1 = 0 # Flag saying that there is no post-processing file saved previously

#
if (postproc_exists == 1) and (postproc_exists_CPU1 == 1):
    STRUCTURES_data = dict_merge(STRUCTURES_data, STRUCTURES_data_CPU1)
elif (postproc_exists != 1) and (postproc_exists_CPU1 == 1):
    STRUCTURES_data = STRUCTURES_data_CPU1
    postproc_exists = 1 # Flag saying that there is a post-processing file

file_postproc_path_CPU2 = postproc_dir+'/'+'STRUCTURES_postprocessing_variables_CPU2.p'
# Flag saying post-processing file for this CPU exists:
try:
    # Try to load the previous post-processing file for this CPU with all information:
    readfile_postproc_CPU2 = open(file_postproc_path_CPU2, 'rb')
    STRUCTURES_data_CPU2 = pickle.load(readfile_postproc_CPU2)
    readfile_postproc_CPU2.close()
    postproc_exists_CPU2 = 1
except Exception, e:
    postproc_exists_CPU2 = 0 # Flag saying that there is no post-processing file saved previously

#
if (postproc_exists == 1) and (postproc_exists_CPU2 == 1):
    STRUCTURES_data = dict_merge(STRUCTURES_data, STRUCTURES_data_CPU2)
elif (postproc_exists != 1) and (postproc_exists_CPU2 == 1):
    STRUCTURES_data = STRUCTURES_data_CPU2
    postproc_exists = 1 # Flag saying that there is a post-processing file

file_postproc_path_CPU3 = postproc_dir+'/'+'STRUCTURES_postprocessing_variables_CPU3.p'
# Flag saying post-processing file for this CPU exists:
try:
    # Try to load the previous post-processing file for this CPU with all information:
    readfile_postproc_CPU3 = open(file_postproc_path_CPU3, 'rb')
    STRUCTURES_data_CPU3 = pickle.load(readfile_postproc_CPU3)
    readfile_postproc_CPU3.close()
    postproc_exists_CPU3 = 1
except Exception, e:
    postproc_exists_CPU3 = 0 # Flag saying that there is no post-processing file saved previously

#
if (postproc_exists == 1) and (postproc_exists_CPU3 == 1):
    STRUCTURES_data = dict_merge(STRUCTURES_data, STRUCTURES_data_CPU3)
elif (postproc_exists != 1) and (postproc_exists_CPU3 == 1):
    STRUCTURES_data = STRUCTURES_data_CPU3
    postproc_exists = 1 # Flag saying that there is a post-processing file

file_postproc_path_CPU4 = postproc_dir+'/'+'STRUCTURES_postprocessing_variables_CPU4.p'
# Flag saying post-processing file for this CPU exists:
try:
    # Try to load the previous post-processing file for this CPU with all information:
    readfile_postproc_CPU4 = open(file_postproc_path_CPU4, 'rb')
    STRUCTURES_data_CPU4 = pickle.load(readfile_postproc_CPU4)
    readfile_postproc_CPU4.close()
    postproc_exists_CPU4 = 1
except Exception, e:
    postproc_exists_CPU4 = 0 # Flag saying that there is no post-processing file saved previously

#
if (postproc_exists == 1) and (postproc_exists_CPU4 == 1):
    STRUCTURES_data = dict_merge(STRUCTURES_data, STRUCTURES_data_CPU4)
elif (postproc_exists != 1) and (postproc_exists_CPU4 == 1):
    STRUCTURES_data = STRUCTURES_data_CPU4
    postproc_exists = 1 # Flag saying that there is a post-processing file

file_postproc_path_CPU5 = postproc_dir+'/'+'STRUCTURES_postprocessing_variables_CPU5.p'
# Flag saying post-processing file for this CPU exists:
try:
    # Try to load the previous post-processing file for this CPU with all information:
    readfile_postproc_CPU5 = open(file_postproc_path_CPU5, 'rb')
    STRUCTURES_data_CPU5 = pickle.load(readfile_postproc_CPU5)
    readfile_postproc_CPU5.close()
    postproc_exists_CPU5 = 1
except Exception, e:
    postproc_exists_CPU5 = 0 # Flag saying that there is no post-processing file saved previously

#
if (postproc_exists == 1) and (postproc_exists_CPU5 == 1):
    STRUCTURES_data = dict_merge(STRUCTURES_data, STRUCTURES_data_CPU5)
elif (postproc_exists != 1) and (postproc_exists_CPU5 == 1):
    STRUCTURES_data = STRUCTURES_data_CPU5
    postproc_exists = 1 # Flag saying that there is a post-processing file

file_postproc_path_CPU6 = postproc_dir+'/'+'STRUCTURES_postprocessing_variables_CPU6.p'
# Flag saying post-processing file for this CPU exists:
try:
    # Try to load the previous post-processing file for this CPU with all information:
    readfile_postproc_CPU6 = open(file_postproc_path_CPU6, 'rb')
    STRUCTURES_data_CPU6 = pickle.load(readfile_postproc_CPU6)
    readfile_postproc_CPU6.close()
    postproc_exists_CPU6 = 1
except Exception, e:
    postproc_exists_CPU6 = 0 # Flag saying that there is no post-processing file saved previously

#
if (postproc_exists == 1) and (postproc_exists_CPU6 == 1):
    STRUCTURES_data = dict_merge(STRUCTURES_data, STRUCTURES_data_CPU6)
elif (postproc_exists != 1) and (postproc_exists_CPU6 == 1):
    STRUCTURES_data = STRUCTURES_data_CPU6
    postproc_exists = 1 # Flag saying that there is a post-processing file

file_postproc_path_CPU7 = postproc_dir+'/'+'STRUCTURES_postprocessing_variables_CPU7.p'
# Flag saying post-processing file for this CPU exists:
try:
    # Try to load the previous post-processing file for this CPU with all information:
    readfile_postproc_CPU7 = open(file_postproc_path_CPU7, 'rb')
    STRUCTURES_data_CPU7 = pickle.load(readfile_postproc_CPU7)
    readfile_postproc_CPU7.close()
    postproc_exists_CPU7 = 1
except Exception, e:
    postproc_exists_CPU7 = 0 # Flag saying that there is no post-processing file saved previously

#
if (postproc_exists == 1) and (postproc_exists_CPU7 == 1):
    STRUCTURES_data = dict_merge(STRUCTURES_data, STRUCTURES_data_CPU7)
elif (postproc_exists != 1) and (postproc_exists_CPU7 == 1):
    STRUCTURES_data = STRUCTURES_data_CPU7
    postproc_exists = 1 # Flag saying that there is a post-processing file

file_postproc_path_CPU8 = postproc_dir+'/'+'STRUCTURES_postprocessing_variables_CPU8.p'
# Flag saying post-processing file for this CPU exists:
try:
    # Try to load the previous post-processing file for this CPU with all information:
    readfile_postproc_CPU8 = open(file_postproc_path_CPU8, 'rb')
    STRUCTURES_data_CPU8 = pickle.load(readfile_postproc_CPU8)
    readfile_postproc_CPU8.close()
    postproc_exists_CPU8 = 1
except Exception, e:
    postproc_exists_CPU8 = 0 # Flag saying that there is no post-processing file saved previously

#
if (postproc_exists == 1) and (postproc_exists_CPU8 == 1):
    STRUCTURES_data = dict_merge(STRUCTURES_data, STRUCTURES_data_CPU8)
elif (postproc_exists != 1) and (postproc_exists_CPU8 == 1):
    STRUCTURES_data = STRUCTURES_data_CPU8
    postproc_exists = 1 # Flag saying that there is a post-processing file

# Save post-processing information to pickle file:
writefile_postproc = open(file_postproc_path, 'wb')
pickle.dump(STRUCTURES_data, writefile_postproc)
writefile_postproc.close()

#
