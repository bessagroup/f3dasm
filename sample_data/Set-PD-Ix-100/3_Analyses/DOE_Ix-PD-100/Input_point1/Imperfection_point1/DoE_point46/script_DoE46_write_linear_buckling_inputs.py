#=====================================================================#
#
# Created by M.A. Bessa on 12-Nov-2019 03:18:07
#=====================================================================#
from abaqusConstants import *
from odbAccess import *
import os
import numpy
import collections
#
os.chdir(r'/home/gkus/F3DAS-master/3_Analyses/DOE_Ix-PD-100/Input_point1/Imperfection_point1/DoE_point46')
with open('DoE46_linear_buckle.inp','w') as File:
    File.write('** Include file with mesh of structure:\n')
    File.write('*INCLUDE, INPUT=include_mesh_DoE46.inp\n')
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

# Create job, run it and wait for completion
inputFile_string = '/home/gkus/F3DAS-master/3_Analyses/DOE_Ix-PD-100/Input_point1/Imperfection_point1/DoE_point46'+'/'+'DoE46_linear_buckle.inp'
job=mdb.JobFromInputFile(name='DoE46_linear_buckle', inputFileName=inputFile_string, 
    type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, 
    memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, userSubroutine='', 
    scratch='', resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN, 
    numDomains=1, activateLoadBalancing=False, multiprocessingMode=DEFAULT, 
    numCpus=1)
#
job.submit()
job.waitForCompletion()

#
