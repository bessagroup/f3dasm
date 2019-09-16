%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%  Function to create the main run file for simulations                   %
%                                                                         %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                                                                         %
%                M.A. Bessa - M.A.Bessa@tudelft.nl		          %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function generate_riks_results_supercompressible_mm(DoE_points,Input_points,UserSettings,jDoE,kImperfection,iInput,jDoE_dir,postproc_dir,CPUid,main_dir,abq_inp_file)
%
% RIKS_imperfection = UserSettings.RIKS_imperfection(kImperfection,:);
RIKS_imperfection = DoE_points(jDoE,8);
% RIKS_imperfection = Input_points(iInput,2:3);



load_dof = 3;

if sign(load_dof) > 0
    sign_string = 'p';
else
    sign_string = 'm';
end

static_scriptfile_name = strcat(jDoE_dir,'/script_DoE',num2str(jDoE),'_write_riks_inputs.py');
fid = fopen(static_scriptfile_name,'wt');

fprintf(fid,'#=====================================================================#\n');
fprintf(fid,'#\n');
fprintf(fid,'# Created by M.A. Bessa on %s\n',datestr(now));
fprintf(fid,'#=====================================================================#\n');
fprintf(fid,'from abaqusConstants import *\n');
% fprintf(fid,'from caeModules import *\n');
% fprintf(fid,'from driverUtils import executeOnCaeStartup\n');
fprintf(fid,'from odbAccess import *\n');
fprintf(fid,'import os\n');
fprintf(fid,'import numpy\n');
fprintf(fid,'import collections\n');
fprintf(fid,'from copy import deepcopy\n');
fprintf(fid,'try:\n');
fprintf(fid,'    import cPickle as pickle  # Improve speed\n');
fprintf(fid,'except ValueError:\n');
fprintf(fid,'    import pickle\n');
fprintf(fid,'\n');
fprintf(fid,'def dict_merge(a, b):\n');
fprintf(fid,'    #recursively merges dict''s. not just simple a[''key''] = b[''key''], if\n');
fprintf(fid,'    #both a and b have a key who''s value is a dict then dict_merge is called\n');
fprintf(fid,'    #on both values and the result stored in the returned dictionary.\n');
fprintf(fid,'    if not isinstance(b, dict):\n');
fprintf(fid,'        return b\n');
fprintf(fid,'    result = deepcopy(a)\n');
% fprintf(fid,'    result = a\n');
fprintf(fid,'    for k, v in b.iteritems():\n');
fprintf(fid,'        if k in result and isinstance(result[k], dict):\n');
fprintf(fid,'                result[k] = dict_merge(result[k], v)\n');
fprintf(fid,'        else:\n');
fprintf(fid,'            result[k] = deepcopy(v)\n');
% fprintf(fid,'            result[k] = v\n');
fprintf(fid,'    return result\n');
fprintf(fid,'\n');
fprintf(fid,'#\n');
%
% Select work directory:
fprintf(fid,'os.chdir(r''%s'')\n',jDoE_dir);
%
fprintf(fid,'# Set directory where the post-processing file is\n');
fprintf(fid,'postproc_dir=''%s''\n',postproc_dir);
fprintf(fid,'#\n');
if UserSettings.run_simul_option == 1 % Then the postprocessing file is associated to the CPU number
    fprintf(fid,'file_postproc_path = postproc_dir+''/''+''STRUCTURES_postprocessing_variables_CPU%s.p''\n',num2str(CPUid));
else % Then the postprocessing file generated previously is unique and independent of CPU number
    fprintf(fid,'file_postproc_path = postproc_dir+''/''+''STRUCTURES_postprocessing_variables.p''\n');
end
fprintf(fid,'# Flag saying post-processing file exists:\n');
fprintf(fid,'try:\n');
fprintf(fid,'    # Try to load a previous post-processing file with all information:\n');
fprintf(fid,'    readfile_postproc = open(file_postproc_path, ''rb'')\n');
fprintf(fid,'    STRUCTURES_data = pickle.load(readfile_postproc)\n');
fprintf(fid,'    readfile_postproc.close()\n');
fprintf(fid,'    postproc_exists = 1\n');
fprintf(fid,'except Exception, e:\n');
fprintf(fid,'    postproc_exists = 0 # Flag saying that there is no post-processing file saved previously\n');
fprintf(fid,'    sys.exit(1) # Exit the code because there is nothing left to do!\n');
fprintf(fid,'\n');
fprintf(fid,'#\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Write input file for RIKS analysis:
fprintf(fid,'with open(''DoE%i_riks.inp'',''wb'') as File:\n',jDoE);
fprintf(fid,'    if STRUCTURES_data[''Input%i''][''Imperfection%i''][''DoE%i''][''coilable''][0] == 0:\n',iInput,kImperfection,jDoE);
fprintf(fid,'        sys.exit() # do not bother running the RIKS analysis because the material will not coil...\n');
fprintf(fid,'    #\n');
fprintf(fid,'    File.write(''** Include file with mesh of structure:\\n'')\n');
fprintf(fid,'    File.write(''*INCLUDE, INPUT=include_mesh_DoE%i.inp\\n'')\n',jDoE);
fprintf(fid,'    File.write(''** \\n'')\n');
fprintf(fid,'    File.write(''** INTERACTION PROPERTIES\\n'')\n');
fprintf(fid,'    File.write(''** \\n'')\n');
fprintf(fid,'    File.write(''*SURFACE INTERACTION,NAME=IMP_TARG\\n'')\n');
fprintf(fid,'    File.write(''1.,\\n'')\n');
fprintf(fid,'    File.write(''*Surface Behavior, no separation, pressure-overclosure=HARD\\n'')\n');
fprintf(fid,'    File.write(''***Surface Interaction, name=IntProp-1\\n'')\n');
fprintf(fid,'    File.write(''** \\n'')\n');
fprintf(fid,'    File.write(''** INTERACTIONS\\n'')\n');
fprintf(fid,'    File.write(''** \\n'')\n');
fprintf(fid,'    File.write(''***CONTACT PAIR,INTERACTION=IMP_TARG\\n'')\n');
fprintf(fid,'    File.write(''**longerons-1-1.all_longerons_surface, AnalyticSurf-1-1.rigid_support\\n'')\n');
fprintf(fid,'    File.write(''** Interaction: Int-1\\n'')\n');
fprintf(fid,'    File.write(''*Contact Pair, interaction=IMP_TARG, type=SURFACE TO SURFACE, no thickness\\n'')\n');
fprintf(fid,'    File.write(''longerons-1-1.all_longerons_surface, AnalyticSurf-1-1.rigid_support\\n'')\n');
fprintf(fid,'    File.write(''**\\n'')\n');
fprintf(fid,'    File.write(''** Seed an imperfection:\\n'')\n');
if UserSettings.analysis_option == 1 % Then we did a LINEAR buckling analysis
    fprintf(fid,'    File.write(''*IMPERFECTION, FILE=DoE%i_linear_buckle, STEP=1\\n'')\n',jDoE);
elseif UserSettings.analysis_option == 2 % Then we did a NONLINEAR buckling analysis
    fprintf(fid,'    File.write(''*IMPERFECTION, FILE=DoE%i_nonlinear_buckle, STEP=2\\n'')\n',jDoE);
elseif UserSettings.analysis_option == 3 % Then we did a LINEAR MODAL analysis
    fprintf(fid,'    File.write(''*IMPERFECTION, FILE=DoE%i_frequency, STEP=1\\n'')\n',jDoE);
else
    % Do nothing
end

for i=1:length(RIKS_imperfection)
    fprintf(fid,'    mode_amplitude = %1.5e/STRUCTURES_data[''Input%i''][''Imperfection%i''][''DoE%i''][''maxDisp_%s''][%i]\n',RIKS_imperfection(i),iInput,kImperfection,jDoE,strcat(sign_string,num2str(round(abs(load_dof)))),i);
    fprintf(fid,'    File.write(''%i, '' + str(mode_amplitude) + ''\\n'')\n',i);
%     fprintf(fid,'    File.write(''%i, %1.5e\\n'')\n',i,RIKS_imperfection(i));
end
fprintf(fid,'    File.write(''** \\n'')\n');
fprintf(fid,'    File.write(''** STEP: Step-1\\n'')\n');
fprintf(fid,'    File.write(''** \\n'')\n');
fprintf(fid,'    File.write(''*Step, name=Step-RIKS, nlgeom=YES, inc=400\\n'')\n');
fprintf(fid,'    File.write(''*Static, riks\\n'')\n');
fprintf(fid,'    File.write(''5.0e-2,1.0,,0.5\\n'')\n');
fprintf(fid,'    File.write(''** \\n'')\n');
fprintf(fid,'    File.write(''** BOUNDARY CONDITIONS\\n'')\n');
fprintf(fid,'    File.write(''** \\n'')\n');
fprintf(fid,'    File.write(''** Name: BC_Zminus Type: Displacement/Rotation\\n'')\n');
fprintf(fid,'    File.write(''*Boundary\\n'')\n');
fprintf(fid,'    File.write(''RP_ZmYmXm, 1, 6\\n'')\n');
fprintf(fid,'    File.write(''** Name: BC_Zplus Type: Displacement/Rotation\\n'')\n');
fprintf(fid,'    File.write(''*Boundary, type=displacement\\n'')\n');
fprintf(fid,'    File.write(''RP_ZpYmXm, 3, 3, -%7.5e\\n'')\n',DoE_points(jDoE,6)*Input_points(iInput,7));
fprintf(fid,'    File.write(''** \\n'')\n');
% fprintf(fid,'    File.write(''** LOADS\\n'')\n');
% fprintf(fid,'    File.write(''** \\n'')\n');
% fprintf(fid,'    File.write(''** Name: Applied_Moment   Type: Moment\\n'')\n');
% fprintf(fid,'    File.write(''***Cload\\n'')\n');
% fprintf(fid,'    File.write(''**RP_ZpYmXm, %i, ''+str(%6.2f*STRUCTURES_data[''Input%i''][''DoE%i''][''MOM_%s_crit'']) + ''\\n'')\n',round(abs(load_dof)),(sign(load_dof)*2.0),iInput,jDoE,strcat(sign_string,num2str(round(abs(load_dof)))) );
% fprintf(fid,'    File.write(''** NOTE: the above applied moment has a factor of 2 due to the way the constraint equations between the 2 reference nodes are applied\\n'')\n');
fprintf(fid,'    File.write(''** \\n'')\n');
fprintf(fid,'    File.write(''** OUTPUT REQUESTS\\n'')\n');
% Do not request restart file, otherwise the output file is way too big.
%     fprintf(fid,'    File.write(''** \\n'')\n');
%     fprintf(fid,'    File.write(''*Restart, write, frequency=1\\n'')\n');
fprintf(fid,'    File.write(''** \\n'')\n');
fprintf(fid,'    File.write(''** FIELD OUTPUT: F-Output-1\\n'')\n');
fprintf(fid,'    File.write(''** \\n'')\n');
fprintf(fid,'    File.write(''*Output, field, variable=PRESELECT, frequency=1\\n'')\n');
fprintf(fid,'    File.write(''** \\n'')\n');
fprintf(fid,'    File.write(''** HISTORY OUTPUT: H-Output-2\\n'')\n');
fprintf(fid,'    File.write(''** \\n'')\n');
fprintf(fid,'    File.write(''*Output, history, frequency=1\\n'')\n');
fprintf(fid,'    File.write(''*Node Output, nset=RP_ZmYmXm\\n'')\n');
fprintf(fid,'    File.write(''RF1, RF2, RF3, RM1, RM2, RM3, U1, U2\\n'')\n');
fprintf(fid,'    File.write(''U3, UR1, UR2, UR3\\n'')\n');
fprintf(fid,'    File.write(''** \\n'')\n');
fprintf(fid,'    File.write(''** HISTORY OUTPUT: H-Output-3\\n'')\n');
fprintf(fid,'    File.write(''** \\n'')\n');
fprintf(fid,'    File.write(''*Node Output, nset=RP_ZpYmXm\\n'')\n');
fprintf(fid,'    File.write(''RF1, RF2, RF3, RM1, RM2, RM3, U1, U2\\n'')\n');
fprintf(fid,'    File.write(''U3, UR1, UR2, UR3\\n'')\n');
fprintf(fid,'    File.write(''** \\n'')\n');
fprintf(fid,'    File.write(''** HISTORY OUTPUT: H-Output-1\\n'')\n');
fprintf(fid,'    File.write(''** \\n'')\n');
fprintf(fid,'    File.write(''*Output, history, variable=PRESELECT, frequency=1\\n'')\n');
fprintf(fid,'    File.write(''*End Step\\n'')\n');
fprintf(fid,'\n');
fprintf(fid,'#\n');
%
fclose(fid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Execute Python Script in ABAQUS CAE                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
unix(strcat(UserSettings.abaqus_path,' cae noGUI=',jDoE_dir,'/script_DoE',num2str(jDoE),'_write_riks_inputs.py'));
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run ABAQUS simulation                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if UserSettings.run_simul_option == 1
    disp(' ');
    disp(['Started running RIKS analysis for DoE',num2str(jDoE),' Imperfection',num2str(kImperfection)]); disp(' ');
    % unix(strcat(UserSettings.abaqus_path,' cae noGUI=',jDoE_dir,'/script_DoE',num2str(jDoE),'_write_implicit_static_input.py'));    
    %
    cd(jDoE_dir);
    %     system(string);
    % Run simulations in local desktop
    string = ['abaqus job=',abq_inp_file,'_riks',...
        ' cpus=',num2str(UserSettings.analysis_parameters.ncpus),' double interactive ask_delete=OFF'];
    system(string);
    %
    cd(main_dir);
    %
    disp(' ');
    disp(['RIKS analysis COMPLETED for DoE',num2str(jDoE),' Imperfection',num2str(kImperfection)]);
% end
%
%
% if UserSettings.postproc_option == 1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Create Python File for Post-processing Abaqus simulations               %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % disp(' ');
    % disp('Started creation of Python file for postprocessing');
    %
    riks_scriptfile_name = strcat(postproc_dir,'/script_write_riks_postproc_CPU',num2str(CPUid),'.py');
    fid = fopen(riks_scriptfile_name,'wt');
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
    fprintf(fid,'from copy import deepcopy\n');
    fprintf(fid,'try:\n');
    fprintf(fid,'    import cPickle as pickle  # Improve speed\n');
    fprintf(fid,'except ValueError:\n');
    fprintf(fid,'    import pickle\n');
    fprintf(fid,'\n');
    fprintf(fid,'def dict_merge(a, b):\n');
    fprintf(fid,'    #recursively merges dict''s. not just simple a[''key''] = b[''key''], if\n');
    fprintf(fid,'    #both a and b have a key who''s value is a dict then dict_merge is called\n');
    fprintf(fid,'    #on both values and the result stored in the returned dictionary.\n');
    fprintf(fid,'    if not isinstance(b, dict):\n');
    fprintf(fid,'        return b\n');
    fprintf(fid,'    result = deepcopy(a)\n');
    % fprintf(fid,'    result = a\n');
    fprintf(fid,'    for k, v in b.iteritems():\n');
    fprintf(fid,'        if k in result and isinstance(result[k], dict):\n');
    fprintf(fid,'                result[k] = dict_merge(result[k], v)\n');
    fprintf(fid,'        else:\n');
    fprintf(fid,'            result[k] = deepcopy(v)\n');
    % fprintf(fid,'            result[k] = v\n');
    fprintf(fid,'    return result\n');
    fprintf(fid,'\n');
    fprintf(fid,'#\n');
    %
    % Select work directory:
    fprintf(fid,'os.chdir(r''%s'')\n',jDoE_dir);
    %
    fprintf(fid,'# Set directory where you want the post-processing file to be generated\n');
    fprintf(fid,'postproc_dir=''%s''\n',postproc_dir);
    fprintf(fid,'#\n');
    fprintf(fid,'file_postproc_path = postproc_dir+''/''+''STRUCTURES_postprocessing_variables_CPU%s.p''\n',num2str(CPUid));
    fprintf(fid,'# Flag saying post-processing file exists:\n');
    fprintf(fid,'try:\n');
    fprintf(fid,'    # Try to load a previous post-processing file with all information:\n');
    fprintf(fid,'    readfile_postproc = open(file_postproc_path, ''rb'')\n');
    fprintf(fid,'    STRUCTURES_data_old = pickle.load(readfile_postproc)\n');
    fprintf(fid,'    readfile_postproc.close()\n');
    fprintf(fid,'    postproc_exists = 1\n');
    fprintf(fid,'except Exception, e:\n');
    fprintf(fid,'    postproc_exists = 0 # Flag saying that there is no post-processing file saved previously\n');
    fprintf(fid,'\n');
    fprintf(fid,'#\n');
%
    fprintf(fid,'# Name of the file job\n');
    fprintf(fid,'jobname=''%s_riks''\n',abq_inp_file);
    fprintf(fid,'#\n');
    
    fprintf(fid,'odbfile=jobname+''.odb'' # Define name of this .odb file\n');
    fprintf(fid,'try:\n');
    fprintf(fid,'    # Try to open this odb file\n');
    fprintf(fid,'    MODELodb = openOdb(path=odbfile) # Open .odb file\n');
    fprintf(fid,'except Exception, e:\n');
    fprintf(fid,'    print >> sys.stderr, ''does not exist''\n');
    fprintf(fid,'    print >> sys.stderr, ''Exception: %%s'' %% str(e)\n');
    fprintf(fid,'    sys.exit(1) # Exit the code because there is nothing left to do!\n');
    fprintf(fid,'\n');
    fprintf(fid,'#\n');
    fprintf(fid,'# Determine the number of steps in the output database.\n');
    fprintf(fid,'mySteps = MODELodb.steps\n');
    fprintf(fid,'numSteps = len(mySteps)\n');
    fprintf(fid,'#\n');
    fprintf(fid,'RP_Zplus_nSet = MODELodb.rootAssembly.nodeSets[''RP_ZPYMXM'']\n');
    fprintf(fid,'RP_Zminus_nSet = MODELodb.rootAssembly.nodeSets[''RP_ZMYMXM'']\n');
    fprintf(fid,'entireSTRUCTURE_nSet = MODELodb.rootAssembly.nodeSets['' ALL NODES'']\n');
    fprintf(fid,'entireSTRUCTURE_elSet = MODELodb.rootAssembly.elementSets['' ALL ELEMENTS'']\n');
    %
    fprintf(fid,'#\n');
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
    fprintf(fid,'# Preallocate quantities for speed\n');
    fprintf(fid,'MODELframe = MODELodb.steps[mySteps.keys()[0]].frames[0] # Undeformed config.\n');
    %
    % Preallocate variables
    postproc_variables = {'U', 'UR', 'RF', 'RM'};
    for ivar = 1:length(postproc_variables)
        fprintf(fid,'%s_Field = MODELframe.fieldOutputs[''%s'']\n',postproc_variables{ivar},postproc_variables{ivar});
        fprintf(fid,'%s_RP_Zplus_SubField = %s_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)\n',postproc_variables{ivar},postproc_variables{ivar});
        fprintf(fid,'%s_RP_Zminus_SubField = %s_Field.getSubset(region=RP_Zminus_nSet, position=NODAL)\n',postproc_variables{ivar},postproc_variables{ivar});
        fprintf(fid,'#\n');
        fprintf(fid,'if isinstance(%s_RP_Zplus_SubField.values[0].data,float):\n',postproc_variables{ivar});
        fprintf(fid,'    # Then variable is a scalar\n');
        fprintf(fid,'    %s_RP_Zplus = numpy.zeros(( totalNumFrames ))\n',postproc_variables{ivar});
        fprintf(fid,'    %s_RP_Zminus = numpy.zeros(( totalNumFrames ))\n',postproc_variables{ivar});
        fprintf(fid,'else:\n');
        fprintf(fid,'    # Variable is an array\n');
        fprintf(fid,'    %s_RP_Zplus = numpy.zeros(( totalNumFrames,len(%s_RP_Zplus_SubField.values[0].data) ))\n',postproc_variables{ivar},postproc_variables{ivar});
        fprintf(fid,'    %s_RP_Zminus = numpy.zeros(( totalNumFrames,len(%s_RP_Zminus_SubField.values[0].data) ))\n',postproc_variables{ivar},postproc_variables{ivar});
        fprintf(fid,'\n');
    end
    %
    % Preallocate variables at integration points
%     postproc_variables_int = {};
    postproc_variables_int = UserSettings.postproc_variables_int;
    for ivar = 1:length(postproc_variables_int)
        fprintf(fid,'%s_Field = MODELframe.fieldOutputs[''%s'']\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        fprintf(fid,'%s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_elSet, position=INTEGRATION_POINT)\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        fprintf(fid,'%s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_elSet, position=INTEGRATION_POINT)\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        fprintf(fid,'#\n');
        if UserSettings.postproc_compute_max == 1
            fprintf(fid,'if isinstance(%s_SubField.values[0].data,float):\n',postproc_variables_int{ivar});
            fprintf(fid,'    # Then variable is a scalar\n');
            fprintf(fid,'    max_%s = numpy.ones(( totalNumFrames ))*(-1e20)\n',postproc_variables_int{ivar});
            fprintf(fid,'else:\n');
            fprintf(fid,'    # Variable is an array\n');
            fprintf(fid,'    max_%s = numpy.ones(( totalNumFrames,len(%s_SubField.values[0].data) ))*(-1e20)\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        end
        if UserSettings.postproc_compute_min == 1
            fprintf(fid,'if isinstance(%s_SubField.values[0].data,float):\n',postproc_variables_int{ivar});
            fprintf(fid,'    # Then variable is a scalar\n');
            fprintf(fid,'    min_%s = numpy.ones(( totalNumFrames ))*(1e20)\n',postproc_variables_int{ivar});
            fprintf(fid,'else:\n');
            fprintf(fid,'    # Variable is an array\n');
            fprintf(fid,'    min_%s = numpy.ones(( totalNumFrames,len(%s_SubField.values[0].data) ))*(1e20)\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        end
        if UserSettings.postproc_compute_av == 1
            fprintf(fid,'if isinstance(%s_SubField.values[0].data,float):\n',postproc_variables_int{ivar});
            fprintf(fid,'    # Then variable is a scalar\n');
            fprintf(fid,'    av_%s = numpy.zeros(( totalNumFrames ))\n',postproc_variables_int{ivar});
            fprintf(fid,'else:\n');
            fprintf(fid,'    # Variable is an array\n');
            fprintf(fid,'    av_%s = numpy.zeros(( totalNumFrames,len(%s_SubField.values[0].data) ))\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        end
        fprintf(fid,'\n');
    end
    %
    % Extract values of variables
    fprintf(fid,'# Loop over the Frames of this Step to extract values of variables\n');
    fprintf(fid,'stepTotalTime = numpy.zeros(numSteps)\n');
    fprintf(fid,'previousFrame = 0\n');
    fprintf(fid,'numFrames = 0\n');
    %
    %     fprintf(fid,'for iStep in range(numSteps):\n');
    %     fprintf(fid,'    previousFrame = previousFrame + numFrames\n');
    fprintf(fid,'stepKey = mySteps.keys()[0]\n');
    fprintf(fid,'step = mySteps[stepKey]\n');
    fprintf(fid,'stepTotalTime = step.timePeriod\n');
    fprintf(fid,'numFrames = len(step.frames)\n');
%     fprintf(fid,'U = numpy.zeros(( totalNumFrames ))\n');
%     fprintf(fid,'RF = numpy.zeros(( totalNumFrames ))\n');
%     fprintf(fid,'maxLE11 = numpy.zeros(( totalNumFrames ))\n');
    fprintf(fid,'#\n');
    fprintf(fid,'for iFrame_step in range (0,numFrames):\n');
    fprintf(fid,'    MODELframe = MODELodb.steps[stepKey].frames[iFrame_step]\n');
    fprintf(fid,'    #\n');
    % Extract values of variables
    for ivar = 1:length(postproc_variables)
        fprintf(fid,'    # Variable: %s\n',postproc_variables{ivar});
        fprintf(fid,'    %s_Field = MODELframe.fieldOutputs[''%s'']\n',postproc_variables{ivar},postproc_variables{ivar});
        fprintf(fid,'    %s_RP_Zplus_SubField = %s_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)\n',postproc_variables{ivar},postproc_variables{ivar});
        fprintf(fid,'    %s_RP_Zminus_SubField = %s_Field.getSubset(region=RP_Zminus_nSet, position=NODAL)\n',postproc_variables{ivar},postproc_variables{ivar});
        fprintf(fid,'    #\n');
        fprintf(fid,'    if isinstance(%s_RP_Zplus_SubField.values[0].data,float):\n',postproc_variables{ivar});
        fprintf(fid,'        # Then variable is a scalar:\n');
        fprintf(fid,'        %s_RP_Zplus[iFrame_step] = %s_RP_Zplus_SubField.values[0].data\n',postproc_variables{ivar},postproc_variables{ivar});
        fprintf(fid,'        %s_RP_Zminus[iFrame_step] = %s_RP_Zminus_SubField.values[0].data\n',postproc_variables{ivar},postproc_variables{ivar});
        fprintf(fid,'        #\n');
        fprintf(fid,'    else:\n');
        fprintf(fid,'        # Variable is an array:\n');
        fprintf(fid,'        for j in range(0, len(%s_RP_Zplus_SubField.values[0].data)):\n',postproc_variables{ivar});
        fprintf(fid,'            %s_RP_Zplus[iFrame_step,j] = %s_RP_Zplus_SubField.values[0].data[j]\n',postproc_variables{ivar},postproc_variables{ivar});
        fprintf(fid,'            %s_RP_Zminus[iFrame_step,j] = %s_RP_Zminus_SubField.values[0].data[j]\n',postproc_variables{ivar},postproc_variables{ivar});
        fprintf(fid,'            #\n');
        fprintf(fid,'    #\n');
        fprintf(fid,'    # Finished saving this variable!\n');
    end
    % Now compute the applied moment and total angle:
%     fprintf(fid,'    # Now compute the critical buckling moment and angle:\n');
%     fprintf(fid,'    U[iFrame_step] = %s_RP_Zplus[iFrame_step,%i] \n',postproc_variables{1},round(abs(load_dof))-1);
%     fprintf(fid,'    RF[iFrame_step] = %s_RP_Zplus[iFrame_step,%i] \n',postproc_variables{2},round(abs(load_dof))-1);
    fprintf(fid,'    # Now compute other local quantities:\n');
    %
    % Extract values of variables
    for ivar = 1:length(postproc_variables_int)
        fprintf(fid,'    # Variable: %s\n',postproc_variables_int{ivar});
        fprintf(fid,'    %s_Field = MODELframe.fieldOutputs[''%s'']\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        fprintf(fid,'    %s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_elSet, position=INTEGRATION_POINT)\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        fprintf(fid,'    %s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_elSet, position=INTEGRATION_POINT)\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        fprintf(fid,'    #\n');
        fprintf(fid,'    if isinstance(%s_SubField.values[0].data,float):\n',postproc_variables_int{ivar});
        fprintf(fid,'        # Then variable is a scalar:\n');
        fprintf(fid,'        for strainValue in %s_SubField.values:\n',postproc_variables_int{ivar});
    if UserSettings.postproc_compute_max == 1
        fprintf(fid,'            if (strainValue.data > max_%s[iFrame]):\n',postproc_variables_int{ivar});
        fprintf(fid,'                max_%s[iFrame_step] = strainValue.data[0]\n',postproc_variables_int{ivar});
        fprintf(fid,'            #\n');
    end
    if UserSettings.postproc_compute_min == 1
        fprintf(fid,'            if (strainValue.data < min_%s[iFrame]):\n',postproc_variables_int{ivar});
        fprintf(fid,'                min_%s[iFrame_step] = strainValue.data[0]\n',postproc_variables_int{ivar});
        fprintf(fid,'            #\n');
    end
    if UserSettings.postproc_compute_av == 1
        fprintf(fid,'            av_%s[iFrame_step] = av_%s[iFrame_step] + strainValue.data[0]\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        fprintf(fid,'            #\n');
        fprintf(fid,'        av_%s[iFrame_step] = av_%s[iFrame_step]/len(%s_SubField.values)\n',postproc_variables_int{ivar},postproc_variables_int{ivar},postproc_variables_int{ivar});
    end
    % ELSE IT IS AN ARRAY:
        fprintf(fid,'    else:\n');
        fprintf(fid,'        # Variable is an array:\n');
        fprintf(fid,'        for strainValue in %s_SubField.values:\n',postproc_variables_int{ivar});
    if UserSettings.postproc_compute_max == 1
        fprintf(fid,'            for j in range(0, len(%s_SubField.values[0].data)):\n',postproc_variables_int{ivar});
        fprintf(fid,'                if (strainValue.data[j] > max_%s[iFrame_step,j]):\n',postproc_variables_int{ivar});
        fprintf(fid,'                    max_%s[iFrame_step,j] = strainValue.data[j]\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        fprintf(fid,'                #\n');
    end
    if UserSettings.postproc_compute_min == 1
        fprintf(fid,'            for j in range(0, len(%s_SubField.values[0].data)):\n',postproc_variables_int{ivar});
        fprintf(fid,'                if (strainValue.data[j] < min_%s[iFrame_step,j]):\n',postproc_variables_int{ivar});
        fprintf(fid,'                    min_%s[iFrame_step,j] = strainValue.data[j]\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        fprintf(fid,'                #\n');
    end
    if UserSettings.postproc_compute_av == 1
        fprintf(fid,'            for j in range(0, len(%s_SubField.values[0].data)):\n',postproc_variables_int{ivar});
        fprintf(fid,'                av_%s[iFrame_step,j] = av_%s[iFrame_step,j] + strainValue.data[j]\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        fprintf(fid,'            #\n');
        fprintf(fid,'        #end for strainValue\n');
        fprintf(fid,'        for j in range(0, len(%s_SubField.values[0].data)):\n',postproc_variables_int{ivar});
        fprintf(fid,'            av_%s[iFrame_step,j] = av_%s[iFrame_step,j]/len(%s_SubField.values)\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        fprintf(fid,'        #\n');
    end 
        fprintf(fid,'    #\n');
        fprintf(fid,'    # Finished saving this variable!\n');
    end
%     % Now compute the applied moment and total angle:
%     fprintf(fid,'    # Now compute the MAXIMUM log strain 11 (LE11):\n');
%     fprintf(fid,'    maxLE11[iFrame_step] = max_%s[iFrame_step,0] \n',postproc_variables_int{1});


    % Merge all the strings to save in dictionary:
    dictionary_string = '';

    % First consider quantities at the INTEGRATION POINTS:
    for ivar = 1:length(postproc_variables_int)
        if UserSettings.postproc_compute_max == 1
            add_string = ['''','riks_max_',postproc_variables_int{ivar},''':max_',postproc_variables_int{ivar}];
            % Add variables to dictionary
            if isempty(dictionary_string) == 1
                dictionary_string = strcat(dictionary_string,add_string);
            else
                dictionary_string = strcat(dictionary_string,', ',add_string);
            end
        end
        if UserSettings.postproc_compute_min == 1
            add_string = ['''','riks_min_',postproc_variables_int{ivar},''':min_',postproc_variables_int{ivar}];
            % Add variables to dictionary
            if isempty(dictionary_string) == 1
                dictionary_string = strcat(dictionary_string,add_string);
            else
                dictionary_string = strcat(dictionary_string,', ',add_string);
            end
        end
        if UserSettings.postproc_compute_av == 1
            add_string = ['''','riks_av_',postproc_variables_int{ivar},''':av_',postproc_variables_int{ivar}];
            % Add variables to dictionary
            if isempty(dictionary_string) == 1
                dictionary_string = strcat(dictionary_string,add_string);
            else
                dictionary_string = strcat(dictionary_string,', ',add_string);
            end
        end
    end
%     % Then consider quantities at the WHOLE ELEMENT:
%     for ivar = 1:length(postproc_variables_whole)
%         if UserSettings.postproc_compute_max == 1
%             add_string = ['''','riks_max_',postproc_variables_whole{ivar},''':max_',postproc_variables_whole{ivar}];
%             % Add variables to dictionary
%             if isempty(dictionary_string) == 1
%                 dictionary_string = strcat(dictionary_string,add_string);
%             else
%                 dictionary_string = strcat(dictionary_string,', ',add_string);
%             end
%         end
%         if UserSettings.postproc_compute_min == 1
%             add_string = ['''','riks_min_',postproc_variables_whole{ivar},''':min_',postproc_variables_whole{ivar}];
%             % Add variables to dictionary
%             if isempty(dictionary_string) == 1
%                 dictionary_string = strcat(dictionary_string,add_string);
%             else
%                 dictionary_string = strcat(dictionary_string,', ',add_string);
%             end
%         end
%         if UserSettings.postproc_compute_av == 1
%             add_string = ['''','riks_av_',postproc_variables_whole{ivar},''':av_',postproc_variables_whole{ivar}];
%             % Add variables to dictionary
%             if isempty(dictionary_string) == 1
%                 dictionary_string = strcat(dictionary_string,add_string);
%             else
%                 dictionary_string = strcat(dictionary_string,', ',add_string);
%             end
%         end
%     end
%     % Then consider quantities at the NODES:
%     for ivar = 1:length(postproc_variables_nodes)
%         if UserSettings.postproc_compute_max == 1
%             add_string = ['''','riks_max_',postproc_variables_nodes{ivar},''':max_',postproc_variables_nodes{ivar}];
%             % Add variables to dictionary
%             if isempty(dictionary_string) == 1
%                 dictionary_string = strcat(dictionary_string,add_string);
%             else
%                 dictionary_string = strcat(dictionary_string,', ',add_string);
%             end
%         end
%         if UserSettings.postproc_compute_min == 1
%             add_string = ['''','riks_min_',postproc_variables_nodes{ivar},''':min_',postproc_variables_nodes{ivar}];
%             % Add variables to dictionary
%             if isempty(dictionary_string) == 1
%                 dictionary_string = strcat(dictionary_string,add_string);
%             else
%                 dictionary_string = strcat(dictionary_string,', ',add_string);
%             end
%         end
%         if UserSettings.postproc_compute_av == 1
%             add_string = ['''','riks_av_',postproc_variables_nodes{ivar},''':av_',postproc_variables_nodes{ivar}];
%             % Add variables to dictionary
%             if isempty(dictionary_string) == 1
%                 dictionary_string = strcat(dictionary_string,add_string);
%             else
%                 dictionary_string = strcat(dictionary_string,', ',add_string);
%             end
%         end
%     end
    %
    % Save variables in database
    fprintf(fid,'\n');
    fprintf(fid,'#\n');
    %
    fprintf(fid,'# Save variables of interest to a database\n');
    % Now take care of the variables that are always outputted, independent of
    % user choice:
    
    for ivar = 1:length(postproc_variables)
        add_string = ['''','riks_RP_Zplus_',postproc_variables{ivar},''':',postproc_variables{ivar},'_RP_Zplus'];
        % Add variables to dictionary
        if isempty(dictionary_string) == 1
            dictionary_string = strcat(dictionary_string,add_string);
        else
            dictionary_string = strcat(dictionary_string,', ',add_string);
        end
    end
    
%     permanent_vars = ['''','riks_U_',sign_string,num2str(round(abs(load_dof))),'''',':U',', ',...
%         '''','riks_RF_',sign_string,num2str(round(abs(load_dof))),'''',':RF'];%,', ',...
%         '''','riks_maxLE11_',sign_string,num2str(round(abs(load_dof))),'''',':maxLE11'];
%     permanent_vars = '''RF_crit'':RF_crit, ''U'':U, ''RF'':RF';
%     permanent_vars = ['''','RF_',sign_string,num2str(round(abs(load_dof))),'_crit','''',':RF_crit, ','''','U_',sign_string,num2str(round(abs(load_dof))),'''',':U, ','''','RF_',sign_string,num2str(round(abs(load_dof))),'''',':RF'];
    %
    
    %
    fprintf(fid,'STRUCTURE_variables = {%s}\n',dictionary_string);
    fprintf(fid,'#\n');
    fprintf(fid,'stringDoE = ''DoE''+str(%i)\n',jDoE);
    fprintf(fid,'stringImperfection = ''Imperfection''+str(%i)\n',kImperfection);
    fprintf(fid,'stringInput = ''Input''+str(%i)\n',iInput);
    fprintf(fid,'STRUCTURES_data_update = {stringInput : {stringImperfection : {stringDoE : STRUCTURE_variables } } }\n');
    fprintf(fid,'if postproc_exists == 1:\n');
    fprintf(fid,'    STRUCTURES_data = dict_merge(STRUCTURES_data_old, STRUCTURES_data_update)\n');
    fprintf(fid,'else:\n');
    fprintf(fid,'    STRUCTURES_data = STRUCTURES_data_update\n');
    fprintf(fid,'\n');
    fprintf(fid,'# Save post-processing information to pickle file:\n');
    fprintf(fid,'writefile_postproc = open(file_postproc_path, ''wb'')\n');
    fprintf(fid,'pickle.dump(STRUCTURES_data, writefile_postproc)\n');
    fprintf(fid,'writefile_postproc.close()\n');
    fprintf(fid,'\n');
    fprintf(fid,'#\n');
    fprintf(fid,'# End of file.\n');
    %
    %
    fclose(fid);
    %
    % disp(' ');
    % disp('Creation of Post-Processing Python file COMPLETED');
    % disp('Elapsed Time [min]: ');
    % disp(toc/60);
    %
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Execute Python Script in ABAQUS CAE                                     %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    disp(' ');
    disp(['Started Abaqus odb post-processing for DoE',num2str(jDoE),' Imperfection',num2str(kImperfection)]);
    disp(' ');
    unix(strcat(UserSettings.abaqus_path,' cae noGUI=',postproc_dir,'/script_write_riks_postproc_CPU',num2str(CPUid),'.py'));
    disp(' ');
    disp(['STRUCTURES_postprocessing_variables.p file UPDATED for DoE',num2str(jDoE),' Imperfection',num2str(kImperfection)]);

%     disp('Elapsed Time [min]: ');
%     disp(toc/60);
    %
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of Function                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
