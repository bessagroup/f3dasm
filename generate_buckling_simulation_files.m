%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%  Generate input files for FEA of the RVEs with elliptical particles     %
%                                                                         %
%  This function is called by the code: ICME_main_code.m                  %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                    Miguel A. Bessa - mbessa@u.northwestern.edu          %
%                             January 2015                                %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function generate_buckling_simulation_files(DoE_points,Input_points,UserSettings,jDoE,kImperfection,iInput,jDoE_dir,main_dir,abq_inp_file)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definition of Global Variables                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% global jDoE_dir UserSettings.abaqus_path;
% global jDoE;
% global abq_inp_file;
% global UserSettings.analysis_parameters.init_time_inc UserSettings.analysis_parameters.total_time UserSettings.analysis_parameters.min_time_inc UserSettings.analysis_parameters.max_time_inc;
% global UserSettings.analysis_parameters.output_time_interval UserSettings.analysis_option;
% global load_dof;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create Python File for Abaqus CAE                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
load_dof = Input_points(iInput,1);

if UserSettings.analysis_option == 1
    string = 'linear_buckling';
elseif UserSettings.analysis_option == 2
    string = 'nonlinear_buckling';
elseif UserSettings.analysis_option == 3
    string = 'frequency';
else
    error('Invalid analysis_option');
end

%% BUCKLING ANALYSIS:
% if UserSettings.analysis_option == 1 % Then create files for LINEAR buckling analysis
    %
%     disp(' ');
%     disp('Started creation of Python file for writing STATIC analysis');
    %
    static_scriptfile_name = strcat(jDoE_dir,'/script_DoE',num2str(jDoE),'_write_',string,'_inputs.py');
    fid = fopen(static_scriptfile_name,'wt');
    %
    fprintf(fid,'#=====================================================================#\n');
    fprintf(fid,'#\n');
    fprintf(fid,'# Created by M.A. Bessa on %s\n',datestr(now));
    fprintf(fid,'#=====================================================================#\n');
    % fprintf(fid,'from abaqus import *\n');
    fprintf(fid,'from abaqusConstants import *\n');
    % fprintf(fid,'from caeModules import *\n');
    % fprintf(fid,'from driverUtils import executeOnCaeStartup\n');
    fprintf(fid,'from odbAccess import *\n');
    fprintf(fid,'import os\n');
    fprintf(fid,'import numpy\n');
    fprintf(fid,'import collections\n');
%     fprintf(fid,'from copy import deepcopy\n');
%     fprintf(fid,'try:\n');
%     fprintf(fid,'    import cPickle as pickle  # Improve speed\n');
%     fprintf(fid,'except ValueError:\n');
%     fprintf(fid,'    import pickle\n');
%     fprintf(fid,'\n');
%     fprintf(fid,'def dict_merge(a, b):\n');
%     fprintf(fid,'    #recursively merges dict''s. not just simple a[''key''] = b[''key''], if\n');
%     fprintf(fid,'    #both a and b have a key who''s value is a dict then dict_merge is called\n');
%     fprintf(fid,'    #on both values and the result stored in the returned dictionary.\n');
%     fprintf(fid,'    if not isinstance(b, dict):\n');
%     fprintf(fid,'        return b\n');
%     fprintf(fid,'    result = deepcopy(a)\n');
%     % fprintf(fid,'    result = a\n');
%     fprintf(fid,'    for k, v in b.iteritems():\n');
%     fprintf(fid,'        if k in result and isinstance(result[k], dict):\n');
%     fprintf(fid,'                result[k] = dict_merge(result[k], v)\n');
%     fprintf(fid,'        else:\n');
%     fprintf(fid,'            result[k] = deepcopy(v)\n');
%     % fprintf(fid,'            result[k] = v\n');
%     fprintf(fid,'    return result\n');
%     fprintf(fid,'\n');
    fprintf(fid,'#\n');
    %
    % Select work directory:
    fprintf(fid,'os.chdir(r''%s'')\n',jDoE_dir);
    %
    % Write input file for LINEAR buckle analysis:
    fprintf(fid,'with open(''DoE%i_linear_buckle.inp'',''w'') as File:\n',jDoE);
    fprintf(fid,'    File.write(''** Include file with mesh of structure:\\n'')\n');
    fprintf(fid,'    File.write(''*INCLUDE, INPUT=include_mesh_DoE%i.inp\\n'')\n',jDoE);
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''** STEP: Step-1\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
%     fprintf(fid,'  File.write(''*Step, name=Step-1, nlgeom=NO, perturbation\\n'')\n');
    fprintf(fid,'    File.write(''*Step, name=Step-1\\n'')\n');
    fprintf(fid,'    File.write(''*Buckle, eigensolver=lanczos\\n'')\n');
    fprintf(fid,'    File.write(''20, 0., , , \\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''** BOUNDARY CONDITIONS\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''** Name: BC_Zminus Type: Displacement/Rotation\\n'')\n');
    fprintf(fid,'    File.write(''*Boundary\\n'')\n');
    fprintf(fid,'    File.write(''RP_ZmYmXm, 1, 6\\n'')\n');
%     fprintf(fid,'    File.write(''** Name: BC_Zplus Type: Displacement/Rotation\\n'')\n');
%     fprintf(fid,'    File.write(''*Boundary\\n'')\n');
%     fprintf(fid,'    File.write(''RP_ZpYmXm, 1, 1\\n'')\n');
%     fprintf(fid,'    File.write(''RP_ZpYmXm, 2, 2\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''** LOADS\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''** Name: Applied_Moment   Type: Moment\\n'')\n');
    fprintf(fid,'    File.write(''*Cload\\n'')\n');
    fprintf(fid,'    File.write(''RP_ZpYmXm, 3, -1.00\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''*Node File\\n'')\n');
    fprintf(fid,'    File.write(''U \\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''*EL PRINT,FREQUENCY=1\\n'')\n');
    fprintf(fid,'    File.write(''*NODE PRINT,FREQUENCY=1\\n'')\n');
    fprintf(fid,'    File.write(''*MODAL FILE\\n'')\n');
    fprintf(fid,'    File.write(''*OUTPUT,FIELD,VAR=PRESELECT\\n'')\n');
    fprintf(fid,'    File.write(''*OUTPUT,HISTORY,FREQUENCY=1\\n'')\n');
    fprintf(fid,'    File.write(''*MODAL OUTPUT\\n'')\n');
    fprintf(fid,'    File.write(''*End Step\\n'')\n');
%     fprintf(fid,'File.close()\n');
    %
    % Run the Preliminary Buckle Analysis to estimate critical buckling load:
    fprintf(fid,'\n');
    % fprintf(fid,'a = mdb.models[''Model-1''].rootAssembly\n');
    fprintf(fid,'# Create job, run it and wait for completion\n');
    fprintf(fid,'inputFile_string = ''%s''+''/''+''DoE%i_linear_buckle.inp''\n',jDoE_dir,jDoE);
    fprintf(fid,'job=mdb.JobFromInputFile(name=''DoE%i_linear_buckle'', inputFileName=inputFile_string, \n',jDoE);
    fprintf(fid,'    type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, \n');
    fprintf(fid,'    memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, \n');
    fprintf(fid,'    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, userSubroutine='''', \n');
    fprintf(fid,'    scratch='''', resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN, \n');
    fprintf(fid,'    numDomains=1, activateLoadBalancing=False, multiprocessingMode=DEFAULT, \n');
    fprintf(fid,'    numCpus=1)\n');
    fprintf(fid,'#\n');
    fprintf(fid,'job.submit()\n');
    fprintf(fid,'job.waitForCompletion()\n');
    fprintf(fid,'\n');
    fprintf(fid,'#\n');
    
%% JUST A LINEAR BUCKLING ANALYSIS:
if UserSettings.analysis_option == 1
    %
%     % Open odb file and determine critical load
%     fprintf(fid,'#Import an Abaqus odb\n');
%     fprintf(fid,'MODELodb=openOdb(''DoE%i_linear_buckle''+''.odb'')\n',jDoE);
%     fprintf(fid,'#\n');
%     fprintf(fid,'# Determine the number of steps in the output database.\n');
%     fprintf(fid,'mySteps = MODELodb.steps\n');
%     fprintf(fid,'numSteps = len(mySteps)\n');
%     fprintf(fid,'# For each step, obtain the following:\n');
%     fprintf(fid,'#     1) The step key.\n');
%     fprintf(fid,'#     2) The number of frames in the step.\n');
%     fprintf(fid,'#     3) The increment number of the last frame in the step.\n');
%     fprintf(fid,'#\n');
%     fprintf(fid,'totalNumFrames = 0\n');
%     fprintf(fid,'for iStep in range(numSteps):\n');
%     fprintf(fid,'    stepKey = mySteps.keys()[iStep]\n');
%     fprintf(fid,'    step = mySteps[stepKey]\n');
%     fprintf(fid,'    numFrames = len(step.frames)\n');
%     fprintf(fid,'    totalNumFrames = totalNumFrames + numFrames\n');
%     fprintf(fid,'    #\n');
%     fprintf(fid,'\n');
%     fprintf(fid,'# Read critical buckling load\n');
%     fprintf(fid,'MODELframe = MODELodb.steps[mySteps.keys()[0]].frames[1] # Undeformed config.\n');
%     fprintf(fid,'import re\n');
%     fprintf(fid,'match_number = re.compile(''-?\\ *[0-9]+\\.?[0-9]*(?:[Ee]\\ *-?\\ *[0-9]+)?'')\n');
%     fprintf(fid,'eigenValue = [float(x) for x in re.findall(match_number, MODELframe.description)]\n');
%     fprintf(fid,'#eigenValue[1] is the eigenValue for Mode = eigenValue[0]\n');
    %
    %
    fclose(fid); % Close the previous file and execute it
    %
%     disp(' ');
%     disp('Creation of Python file COMPLETED');
%     disp('Elapsed Time [min]: ');
%     disp(toc/60);
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Execute Python Script in ABAQUS CAE                                     %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    disp(' ');
    disp(['Started LINEAR Buckling analysis of DoE',num2str(jDoE),' Imperfection',num2str(kImperfection)]); disp(' ');
    unix(strcat(UserSettings.abaqus_path,' cae noGUI=',jDoE_dir,'/script_DoE',num2str(jDoE),'_write_',string,'_inputs.py'));
    disp(' ');
    disp(['COMPLETED LINEAR Buckling analysis of DoE',num2str(jDoE),' Imperfection',num2str(kImperfection)]);
%     disp('Elapsed Time [min]: ');
%     disp(toc/60);
    %
    %
%% NONLINEAR BUCKLING ANALYSIS:
elseif UserSettings.analysis_option == 2 % Then create files for NONLINEAR buckling analysis
    %
%     disp(' ');
%     disp('Started creation of Python file for writing STATIC analysis');
    %
%     static_scriptfile_name = strcat(jDoE_dir,'/script_DoE',num2str(jDoE),'_write_implicit_static_input.py');
%     fid = fopen(static_scriptfile_name,'wt');
    %
%     fprintf(fid,'#=====================================================================#\n');
%     fprintf(fid,'#\n');
%     fprintf(fid,'# Created by M.A. Bessa on %s\n',datestr(now));
%     fprintf(fid,'#=====================================================================#\n');
%     % fprintf(fid,'from abaqus import *\n');
%     fprintf(fid,'from abaqusConstants import *\n');
%     % fprintf(fid,'from caeModules import *\n');
%     % fprintf(fid,'from driverUtils import executeOnCaeStartup\n');
%     fprintf(fid,'from odbAccess import *\n');
%     fprintf(fid,'import os\n');
%     fprintf(fid,'import numpy\n');
%     fprintf(fid,'import collections\n');
% %     fprintf(fid,'from copy import deepcopy\n');
% %     fprintf(fid,'try:\n');
% %     fprintf(fid,'    import cPickle as pickle  # Improve speed\n');
% %     fprintf(fid,'except ValueError:\n');
% %     fprintf(fid,'    import pickle\n');
% %     fprintf(fid,'\n');
% %     fprintf(fid,'def dict_merge(a, b):\n');
% %     fprintf(fid,'    #recursively merges dict''s. not just simple a[''key''] = b[''key''], if\n');
% %     fprintf(fid,'    #both a and b have a key who''s value is a dict then dict_merge is called\n');
% %     fprintf(fid,'    #on both values and the result stored in the returned dictionary.\n');
% %     fprintf(fid,'    if not isinstance(b, dict):\n');
% %     fprintf(fid,'        return b\n');
% %     fprintf(fid,'    result = deepcopy(a)\n');
% %     % fprintf(fid,'    result = a\n');
% %     fprintf(fid,'    for k, v in b.iteritems():\n');
% %     fprintf(fid,'        if k in result and isinstance(result[k], dict):\n');
% %     fprintf(fid,'                result[k] = dict_merge(result[k], v)\n');
% %     fprintf(fid,'        else:\n');
% %     fprintf(fid,'            result[k] = deepcopy(v)\n');
% %     % fprintf(fid,'            result[k] = v\n');
% %     fprintf(fid,'    return result\n');
% %     fprintf(fid,'\n');
%     fprintf(fid,'#\n');
%     %
%     % Select work directory:
%     fprintf(fid,'os.chdir(r''%s'')\n',jDoE_dir);
    %
%     % Write input file for LINEAR buckle analysis:
%     fprintf(fid,'with open(''DoE%i_linear_buckle.inp'',''w'') as File:\n',jDoE);
%     fprintf(fid,'    File.write(''** Include file with mesh of structure:\\n'')\n');
%     fprintf(fid,'    File.write(''*INCLUDE, INPUT=include_mesh_DoE%i.inp\\n'')\n',jDoE);
%     fprintf(fid,'    File.write(''** \\n'')\n');
%     fprintf(fid,'    File.write(''** STEP: Step-1\\n'')\n');
%     fprintf(fid,'    File.write(''** \\n'')\n');
% %     fprintf(fid,'  File.write(''*Step, name=Step-1, nlgeom=NO, perturbation\\n'')\n');
%     fprintf(fid,'    File.write(''*Step, name=Step-1\\n'')\n');
%     fprintf(fid,'    File.write(''*Buckle, eigensolver=lanczos\\n'')\n');
%     fprintf(fid,'    File.write(''20, 0., , , \\n'')\n');
%     fprintf(fid,'    File.write(''** \\n'')\n');
%     fprintf(fid,'    File.write(''** BOUNDARY CONDITIONS\\n'')\n');
%     fprintf(fid,'    File.write(''** \\n'')\n');
%     fprintf(fid,'    File.write(''** Name: BC_Zminus Type: Displacement/Rotation\\n'')\n');
%     fprintf(fid,'    File.write(''*Boundary\\n'')\n');
%     fprintf(fid,'    File.write(''RP_Zminus, 1, 1\\n'')\n');
%     fprintf(fid,'    File.write(''RP_Zminus, 2, 2\\n'')\n');
%     fprintf(fid,'    File.write(''RP_Zminus, 3, 3\\n'')\n');
%     if round(abs(load_dof)) ~= 4
%         fprintf(fid,'    File.write(''RP_Zminus, 4, 4\\n'')\n');
%     end
%     if round(abs(load_dof)) ~= 5
%         fprintf(fid,'    File.write(''RP_Zminus, 5, 5\\n'')\n');
%     end
%     if round(abs(load_dof)) ~= 6
%         fprintf(fid,'    File.write(''RP_Zminus, 6, 6\\n'')\n');
%     end
%     fprintf(fid,'    File.write(''** Name: BC_Zplus Type: Displacement/Rotation\\n'')\n');
%     fprintf(fid,'    File.write(''*Boundary\\n'')\n');
%     fprintf(fid,'    File.write(''RP_Zplus, 1, 1\\n'')\n');
%     fprintf(fid,'    File.write(''RP_Zplus, 2, 2\\n'')\n');
%     if round(abs(load_dof)) ~= 4
%         fprintf(fid,'    File.write(''RP_Zplus, 4, 4\\n'')\n');
%     end
%     if round(abs(load_dof)) ~= 5
%         fprintf(fid,'    File.write(''RP_Zplus, 5, 5\\n'')\n');
%     end
%     if round(abs(load_dof)) ~= 6
%         fprintf(fid,'    File.write(''RP_Zplus, 6, 6\\n'')\n');
%     end
%     fprintf(fid,'    File.write(''** \\n'')\n');
%     fprintf(fid,'    File.write(''** LOADS\\n'')\n');
%     fprintf(fid,'    File.write(''** \\n'')\n');
%     fprintf(fid,'    File.write(''** Name: Applied_Moment   Type: Moment\\n'')\n');
%     fprintf(fid,'    File.write(''*Cload\\n'')\n');
%     fprintf(fid,'    File.write(''RP_Zplus, %i, %6.2f\\n'')\n',round(abs(load_dof)),(sign(load_dof)*2.0));
%     fprintf(fid,'    File.write(''** \\n'')\n');
%     fprintf(fid,'    File.write(''*Node File\\n'')\n');
%     fprintf(fid,'    File.write(''U \\n'')\n');
%     fprintf(fid,'    File.write(''** \\n'')\n');
%     fprintf(fid,'    File.write(''*EL PRINT,FREQUENCY=1\\n'')\n');
%     fprintf(fid,'    File.write(''*NODE PRINT,FREQUENCY=1\\n'')\n');
%     fprintf(fid,'    File.write(''*MODAL FILE\\n'')\n');
%     fprintf(fid,'    File.write(''*OUTPUT,FIELD,VAR=PRESELECT\\n'')\n');
%     fprintf(fid,'    File.write(''*OUTPUT,HISTORY,FREQUENCY=1\\n'')\n');
%     fprintf(fid,'    File.write(''*MODAL OUTPUT\\n'')\n');
%     fprintf(fid,'    File.write(''*End Step\\n'')\n');
%     %
%     % Run the Preliminary Buckle Analysis to estimate critical buckling load:
%     fprintf(fid,'File.close()'\n');
%     fprintf(fid,'\n');
%     % fprintf(fid,'a = mdb.models[''Model-1''].rootAssembly\n');
%     fprintf(fid,'# Create job, run it and wait for completion\n');
%     fprintf(fid,'inputFile_string = ''%s''+''/''+''DoE%i_linear_buckle.inp''\n',jDoE_dir,jDoE);
%     fprintf(fid,'job=mdb.JobFromInputFile(name=''DoE%i_linear_buckle'', inputFileName=inputFile_string, \n',jDoE);
%     fprintf(fid,'    type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, \n');
%     fprintf(fid,'    memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, \n');
%     fprintf(fid,'    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, userSubroutine='''', \n');
%     fprintf(fid,'    scratch='''', resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN, \n');
%     fprintf(fid,'    numDomains=1, activateLoadBalancing=False, multiprocessingMode=DEFAULT, \n');
%     fprintf(fid,'    numCpus=1)\n');
%     fprintf(fid,'#\n');
%     fprintf(fid,'job.submit()\n');
%     fprintf(fid,'job.waitForCompletion()\n');
    %
    % Open odb file and determine critical load
    fprintf(fid,'#Import an Abaqus odb\n');
    fprintf(fid,'MODELodb=openOdb(''DoE%i_linear_buckle''+''.odb'')\n',jDoE);
    fprintf(fid,'#\n');
    fprintf(fid,'# Determine the number of steps in the output database.\n');
    fprintf(fid,'mySteps = MODELodb.steps\n');
    fprintf(fid,'numSteps = len(mySteps)\n');
    fprintf(fid,'# For each step, obtain the following:\n');
    fprintf(fid,'#     1) The step key.\n');
    fprintf(fid,'#     2) The number of frames in the step.\n');
    fprintf(fid,'#     3) The increment number of the last frame in the step.\n');
    fprintf(fid,'#\n');
    fprintf(fid,'totalNumFrames = 0\n');
    fprintf(fid,'for iStep in range(numSteps):\n');
    fprintf(fid,'    stepKey = mySteps.keys()[iStep]\n');
    fprintf(fid,'    step = mySteps[stepKey]\n');
    fprintf(fid,'    numFrames = len(step.frames)\n');
    fprintf(fid,'    totalNumFrames = totalNumFrames + numFrames\n');
    fprintf(fid,'    #\n');
    fprintf(fid,'\n');
    fprintf(fid,'# Read critical buckling load\n');
    fprintf(fid,'MODELframe = MODELodb.steps[mySteps.keys()[0]].frames[1] # Undeformed config.\n');
    fprintf(fid,'import re\n');
    fprintf(fid,'match_number = re.compile(''-?\\ *[0-9]+\\.?[0-9]*(?:[Ee]\\ *-?\\ *[0-9]+)?'')\n');
    fprintf(fid,'eigenValue = [float(x) for x in re.findall(match_number, MODELframe.description)]\n');
    fprintf(fid,'#eigenValue[1] is the eigenValue for Mode = eigenValue[0]\n');
    %
    % Write input file for static analysis based on estimated critical load:
    fprintf(fid,'with open(''%s_implicit_static.inp'',''w'') as File:\n',abq_inp_file);
    fprintf(fid,'    File.write(''** Include file with mesh of structure:\\n'')\n');
    fprintf(fid,'    File.write(''*INCLUDE, INPUT=include_mesh_DoE%i.inp\\n'')\n',jDoE);
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''** STEP: Step-1\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''*Step, name=Step-1, nlgeom=YES\\n'')\n');
    fprintf(fid,'    File.write(''*Static\\n'')\n');
    fprintf(fid,'    File.write(''%6.5E,%6.5E,%6.5E,%6.5E\\n'')\n',UserSettings.analysis_parameters.init_time_inc,UserSettings.analysis_parameters.total_time,UserSettings.analysis_parameters.min_time_inc,UserSettings.analysis_parameters.max_time_inc);
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''** BOUNDARY CONDITIONS\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''** Name: BC_Zminus Type: Displacement/Rotation\\n'')\n');
    fprintf(fid,'    File.write(''*Boundary\\n'')\n');
    fprintf(fid,'    File.write(''RP_Zminus, 1, 1\\n'')\n');
    fprintf(fid,'    File.write(''RP_Zminus, 2, 2\\n'')\n');
    fprintf(fid,'    File.write(''RP_Zminus, 3, 3\\n'')\n');
    if round(abs(load_dof)) ~= 4
        fprintf(fid,'    File.write(''RP_Zminus, 4, 4\\n'')\n');
    end
    if round(abs(load_dof)) ~= 5
        fprintf(fid,'    File.write(''RP_Zminus, 5, 5\\n'')\n');
    end
    if round(abs(load_dof)) ~= 6
        fprintf(fid,'    File.write(''RP_Zminus, 6, 6\\n'')\n');
    end
    fprintf(fid,'    File.write(''** Name: BC_Zplus Type: Displacement/Rotation\\n'')\n');
    fprintf(fid,'    File.write(''*Boundary\\n'')\n');
    fprintf(fid,'    File.write(''RP_Zplus, 1, 1\\n'')\n');
    fprintf(fid,'    File.write(''RP_Zplus, 2, 2\\n'')\n');
    if round(abs(load_dof)) ~= 4
        fprintf(fid,'    File.write(''RP_Zplus, 4, 4\\n'')\n');
    end
    if round(abs(load_dof)) ~= 5
        fprintf(fid,'    File.write(''RP_Zplus, 5, 5\\n'')\n');
    end
    if round(abs(load_dof)) ~= 6
        fprintf(fid,'    File.write(''RP_Zplus, 6, 6\\n'')\n');
    end
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''** LOADS\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''** Name: Applied_Moment   Type: Moment\\n'')\n');
    fprintf(fid,'    File.write(''*Cload\\n'')\n');
    fprintf(fid,'    File.write(''RP_Zplus, %i, ''+str((%6.2f*eigenValue[1])*1.2) + ''\\n'')\n',round(abs(load_dof)),(sign(load_dof)*2.0));
    fprintf(fid,'    File.write(''** NOTE 1: the above applied moment has a factor of 2 due to the way the constraint equations between the 2 reference nodes are implemented\\n'')\n');
    fprintf(fid,'    File.write(''** NOTE 2: Also, the above applied moment is 20%% higher than the estimated critical moment (i.e. M_cr*2.4)\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''** OUTPUT REQUESTS\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''*Restart, write, frequency=1\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''** FIELD OUTPUT: F-Output-1\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''*Output, field, variable=PRESELECT\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''** HISTORY OUTPUT: H-Output-2\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''*Output, history\\n'')\n');
    fprintf(fid,'    File.write(''*Node Output, nset=RP_Zminus\\n'')\n');
    fprintf(fid,'    File.write(''RF1, RF2, RF3, RM1, RM2, RM3, U1, U2\\n'')\n');
    fprintf(fid,'    File.write(''U3, UR1, UR2, UR3\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''** HISTORY OUTPUT: H-Output-3\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''*Node Output, nset=RP_Zplus\\n'')\n');
    fprintf(fid,'    File.write(''RF1, RF2, RF3, RM1, RM2, RM3, U1, U2\\n'')\n');
    fprintf(fid,'    File.write(''U3, UR1, UR2, UR3\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''** HISTORY OUTPUT: H-Output-1\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''*Output, history, variable=PRESELECT\\n'')\n');
    fprintf(fid,'    File.write(''*End Step\\n'')\n');
%     fprintf(fid,'File.close()\n');
    fprintf(fid,'\n');
    fprintf(fid,'\n');
    %
    %
    fclose(fid);
    %
%     disp(' ');
%     disp('Creation of Python file COMPLETED');
%     disp('Elapsed Time [min]: ');
%     disp(toc/60);
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Execute Python Script in ABAQUS CAE                                     %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    disp(' ');
    disp(['Started PRELIMINARY LINEAR Buckling analysis of DoE',num2str(jDoE)]); disp(' ');
    unix(strcat(UserSettings.abaqus_path,' cae noGUI=',jDoE_dir,'/script_DoE',num2str(jDoE),'_write_',string,'_inputs.py'));
    disp(' ');
    disp(['COMPLETED PRELIMINARY LINEAR Buckling of DoE',num2str(jDoE)]);
%     disp('Elapsed Time [min]: ');
%     disp(toc/60);
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Run ABAQUS simulation                                                  %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if UserSettings.run_simul_option == 1
        cd(jDoE_dir);
        %     system(string);
        % Run simulations in local desktop
        string = ['abaqus job=',abq_inp_file,'_implicit_static',...
            ' cpus=',num2str(UserSettings.analysis_parameters.ncpus),' double interactive ask_delete=OFF'];
        system(string);
        %
        disp(' ');
        disp(['COMPLETED IMPLICIT STATIC analysis of DoE',num2str(jDoE),' Imperfection',num2str(kImperfection)]);
        %
        cd(main_dir);
    end
    %
elseif UserSettings.analysis_option == 3 % Then create files for FREQUENCY analysis
    %
    %
%     static_scriptfile_name = strcat(jDoE_dir,'/script_DoE',num2str(jDoE),'_write_frequency_inputs.py');
%     fid = fopen(static_scriptfile_name,'wt');
    %
%     fprintf(fid,'#=====================================================================#\n');
%     fprintf(fid,'#\n');
%     fprintf(fid,'# Created by M.A. Bessa on %s\n',datestr(now));
%     fprintf(fid,'#=====================================================================#\n');
%     % fprintf(fid,'from abaqus import *\n');
%     fprintf(fid,'from abaqusConstants import *\n');
%     % fprintf(fid,'from caeModules import *\n');
%     % fprintf(fid,'from driverUtils import executeOnCaeStartup\n');
%     fprintf(fid,'from odbAccess import *\n');
%     fprintf(fid,'import os\n');
%     fprintf(fid,'import numpy\n');
%     fprintf(fid,'import collections\n');
% %     fprintf(fid,'from copy import deepcopy\n');
% %     fprintf(fid,'try:\n');
% %     fprintf(fid,'    import cPickle as pickle  # Improve speed\n');
% %     fprintf(fid,'except ValueError:\n');
% %     fprintf(fid,'    import pickle\n');
% %     fprintf(fid,'\n');
% %     fprintf(fid,'def dict_merge(a, b):\n');
% %     fprintf(fid,'    #recursively merges dict''s. not just simple a[''key''] = b[''key''], if\n');
% %     fprintf(fid,'    #both a and b have a key who''s value is a dict then dict_merge is called\n');
% %     fprintf(fid,'    #on both values and the result stored in the returned dictionary.\n');
% %     fprintf(fid,'    if not isinstance(b, dict):\n');
% %     fprintf(fid,'        return b\n');
% %     fprintf(fid,'    result = deepcopy(a)\n');
% %     % fprintf(fid,'    result = a\n');
% %     fprintf(fid,'    for k, v in b.iteritems():\n');
% %     fprintf(fid,'        if k in result and isinstance(result[k], dict):\n');
% %     fprintf(fid,'                result[k] = dict_merge(result[k], v)\n');
% %     fprintf(fid,'        else:\n');
% %     fprintf(fid,'            result[k] = deepcopy(v)\n');
% %     % fprintf(fid,'            result[k] = v\n');
% %     fprintf(fid,'    return result\n');
% %     fprintf(fid,'\n');
%     fprintf(fid,'#\n');
%     %
%     % Select work directory:
%     fprintf(fid,'os.chdir(r''%s'')\n',jDoE_dir);
    %
    % Write input file for LINEAR buckle analysis:
    fprintf(fid,'with open(''DoE%i_frequency.inp'',''w'') as File:\n',jDoE);
    fprintf(fid,'    File.write(''** Include file with mesh of structure:\\n'')\n');
    fprintf(fid,'    File.write(''*INCLUDE, INPUT=include_mesh_DoE%i.inp\\n'')\n',jDoE);
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''** STEP: Step-1\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
%     fprintf(fid,'  File.write(''*Step, name=Step-1, nlgeom=NO, perturbation\\n'')\n');
    fprintf(fid,'    File.write(''*Step, name=Step-1\\n'')\n');
    fprintf(fid,'    File.write(''*Frequency, eigensolver=lanczos, acoustic coupling=off, normalization=mass\\n'')\n');
    fprintf(fid,'    File.write(''20, 0., , , \\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''** BOUNDARY CONDITIONS\\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''** Name: BC_Zminus Type: Displacement/Rotation\\n'')\n');
    fprintf(fid,'    File.write(''*Boundary\\n'')\n');
    fprintf(fid,'    File.write(''RP_Zminus, 1, 6\\n'')\n'); % ENCASTRE
    fprintf(fid,'    File.write(''** Name: BC_Zplus Type: Displacement/Rotation\\n'')\n');
    fprintf(fid,'    File.write(''*Boundary\\n'')\n');
    fprintf(fid,'    File.write(''RP_Zplus, 4, 6\\n'')\n'); % Do not allow rotations 
    fprintf(fid,'    File.write(''** \\n'')\n');
%     fprintf(fid,'    File.write(''** LOADS\\n'')\n');
%     fprintf(fid,'    File.write(''** \\n'')\n');
%     fprintf(fid,'    File.write(''** Name: Applied_Moment   Type: Moment\\n'')\n');
%     fprintf(fid,'    File.write(''*Cload\\n'')\n');
%     fprintf(fid,'    File.write(''RP_Zplus, %i, %6.2f\\n'')\n',round(abs(load_dof)),(sign(load_dof)*2.0));
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''*Node File\\n'')\n');
    fprintf(fid,'    File.write(''U \\n'')\n');
    fprintf(fid,'    File.write(''** \\n'')\n');
    fprintf(fid,'    File.write(''*EL PRINT,FREQUENCY=1\\n'')\n');
    fprintf(fid,'    File.write(''*NODE PRINT,FREQUENCY=1\\n'')\n');
    fprintf(fid,'    File.write(''*MODAL FILE\\n'')\n');
    fprintf(fid,'    File.write(''*OUTPUT,FIELD,VAR=PRESELECT\\n'')\n');
    fprintf(fid,'    File.write(''*OUTPUT,HISTORY,FREQUENCY=1\\n'')\n');
    fprintf(fid,'    File.write(''*MODAL OUTPUT\\n'')\n');
    fprintf(fid,'    File.write(''*End Step\\n'')\n');
%     fprintf(fid,'File.close()\n');
    %
    % Run the Preliminary Buckle Analysis to estimate critical buckling load:
    fprintf(fid,'\n');
    % fprintf(fid,'a = mdb.models[''Model-1''].rootAssembly\n');
    fprintf(fid,'# Create job, run it and wait for completion\n');
    fprintf(fid,'inputFile_string = ''%s''+''/''+''DoE%i_frequency.inp''\n',jDoE_dir,jDoE);
    fprintf(fid,'job=mdb.JobFromInputFile(name=''DoE%i_frequency'', inputFileName=inputFile_string, \n',jDoE);
    fprintf(fid,'    type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, \n');
    fprintf(fid,'    memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, \n');
    fprintf(fid,'    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, userSubroutine='''', \n');
    fprintf(fid,'    scratch='''', resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN, \n');
    fprintf(fid,'    numDomains=1, activateLoadBalancing=False, multiprocessingMode=DEFAULT, \n');
    fprintf(fid,'    numCpus=1)\n');
    fprintf(fid,'#\n');
    fprintf(fid,'job.submit()\n');
    fprintf(fid,'job.waitForCompletion()\n');
    fprintf(fid,'\n');
    fprintf(fid,'#\n');
    %
    %
    fclose(fid);
    %
%     disp(' ');
%     disp('Creation of Python file COMPLETED');
%     disp('Elapsed Time [min]: ');
%     disp(toc/60);
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Execute Python Script in ABAQUS CAE                                     %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    disp(' ');
    disp(['Started FREQUENCY analysis of DoE',num2str(jDoE)]); disp(' ');
    unix(strcat(UserSettings.abaqus_path,' cae noGUI=',jDoE_dir,'/script_DoE',num2str(jDoE),'_write_',string,'_inputs.py'));
    disp(' ');
    disp(['COMPLETED FREQUENCY analysis of DoE',num2str(jDoE)]);
%     disp('Elapsed Time [min]: ');
%     disp(toc/60);
    %
    %
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of Function                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
end
