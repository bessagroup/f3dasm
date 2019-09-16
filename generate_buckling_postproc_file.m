%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%  Function to create the main run file for simulations                   %
%                                                                         %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                                                                         %
%                                                                         %
%                Miguel Anibal Bessa - mbessa@u.northwestern.edu          %
%                              January 2016                               %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function generate_buckling_postproc_file(DoE_points,Input_points,UserSettings,jDoE,kImperfection,iInput,jDoE_dir,postproc_dir,CPUid)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definition of Global Variables                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% global jDoE_dir jDoE postproc_dir dummy1 dummy2 dir_fea_name nnodes ncpus...
%     pbcs_option UserSettings.analysis_option Lx Ly;
% global jRVE T0_global input_BC strain_option UserSettings.run_simul_option ...
%     UserSettings.abq_inputs_option kSample;
% global matrix_material particle_material interphase_material ...
%     abq_inp_file UserSettings.abaqus_path file_name runfile_name ...
%     UserSettings.postproc_variables_int UserSettings.postproc_variables_whole ...
%     UserSettings.postproc_variables_nodes UserSettings.local_averages;
% global load_dof;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create Python File for Post-processing Abaqus simulations               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% disp(' ');
% disp('Started creation of Python file for postprocessing');
%

% load_dof = Input_points(iInput,1);
load_dof = 3;

if sign(load_dof) > 0
    sign_string = 'p';
else
    sign_string = 'm';
end

static_scriptfile_name = strcat(postproc_dir,'/script_write_postproc_CPU',num2str(CPUid),'.py');
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
fprintf(fid,'import re\n');
% fprintf(fid,'match_number = re.compile(''-?\\ *[0-9]+\\.?[0-9]*(?:[Ee]\\ *-?\\ *[0-9]+)?'')\n');
fprintf(fid,'match_number = re.compile(''-?\\ *[0-9]+\\.?[0-9]*(?:[Ee]\\ *[-+]?\\ *[0-9]+)?'')\n');
%
% Select work directory:
fprintf(fid,'#\n');
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
%
if UserSettings.analysis_option ~= 2 % In these cases we get MOM_crit from LINEAR BUCKLING file
    % TRY to open odb file and determine critical load
    fprintf(fid,'try:\n');
    fprintf(fid,'    # Try to open this odb file\n');
    fprintf(fid,'    #Import Abaqus odb for the Buckle analysis\n');
    fprintf(fid,'    MODELodb=openOdb(''DoE%i_linear_buckle''+''.odb'')\n',jDoE);
    fprintf(fid,'    #\n');
    
    %%%%%%%%%%%%%%%%%%%
    fprintf(fid,'except Exception, e:\n');
    fprintf(fid,'    print >> sys.stderr, ''does not exist''\n');
    fprintf(fid,'    print >> sys.stderr, ''Exception: %%s'' %% str(e)\n');
    fprintf(fid,'    sys.exit(1) # Exit the code because there is nothing left to do!\n');
    fprintf(fid,'#\n');
    fprintf(fid,'\n');
    
    
    fprintf(fid,'# Determine the number of steps in the output database.\n');
    fprintf(fid,'mySteps = MODELodb.steps\n');
    fprintf(fid,'numSteps = len(mySteps)\n');
    fprintf(fid,'stepKey = mySteps.keys()[0]\n');
    fprintf(fid,'step = mySteps[stepKey]\n');
    fprintf(fid,'numFrames = len(step.frames)\n');
    fprintf(fid,'#\n');
    fprintf(fid,'maxDisp = numpy.zeros(( numFrames ))\n');
    fprintf(fid,'RP_Zplus_nSet = MODELodb.rootAssembly.nodeSets[''RP_ZPYMXM'']\n');
    fprintf(fid,'RP_Zminus_nSet = MODELodb.rootAssembly.nodeSets[''RP_ZMYMXM'']\n');
    fprintf(fid,'entireSTRUCTURE_nSet = MODELodb.rootAssembly.nodeSets['' ALL NODES'']\n');
    fprintf(fid,'entireSTRUCTURE_elSet = MODELodb.rootAssembly.elementSets['' ALL ELEMENTS'']\n');
    fprintf(fid,'#\n');
    fprintf(fid,'# Read critical buckling load\n');
%     fprintf(fid,'MODELframe = MODELodb.steps[mySteps.keys()[0]].frames[1]\n');
%     fprintf(fid,'eigenValue = [float(x) for x in re.findall(match_number, MODELframe.description)]\n');
%     fprintf(fid,'#eigenValue[1] is the eigenValue for Mode = eigenValue[0]\n');
%     fprintf(fid,'Pcrit_perturb=eigenValue\n');
%     fprintf(fid,'#\n');
    fprintf(fid,'P_crit=numpy.zeros( numFrames-1 )\n');
    fprintf(fid,'coilable=numpy.zeros( numFrames-1 )\n');
%     fprintf(fid,'Pcrit_perturb=numpy.zeros(( numFrames,2 ))\n');
    fprintf(fid,'for iFrame_step in range(1,numFrames):\n');
    fprintf(fid,'    # Read critical buckling load\n');
    fprintf(fid,'    MODELframe = MODELodb.steps[mySteps.keys()[0]].frames[iFrame_step]\n');
    fprintf(fid,'    eigenValue = [float(x) for x in re.findall(match_number, MODELframe.description)]\n');
    fprintf(fid,'    #eigenValue[1] is the eigenValue for Mode = eigenValue[0]\n');
%     fprintf(fid,'    Pcrit_perturb[iFrame_step,0]=eigenValue[0]\n');
%     fprintf(fid,'    Pcrit_perturb[iFrame_step,1]=eigenValue[1]\n');
    fprintf(fid,'    P_crit[iFrame_step-1]=eigenValue[1]\n');
    fprintf(fid,'    #Now check if this is a coilable mode\n');
    fprintf(fid,'    UR_Field = MODELframe.fieldOutputs[''UR'']\n');
    fprintf(fid,'    UR_SubField = UR_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)\n');
    fprintf(fid,'    UR_Mode1_RP_ZpYmXm = UR_SubField.values[0].data\n');
    fprintf(fid,'    U_Field = MODELframe.fieldOutputs[''U'']\n');
    fprintf(fid,'    U_SubField = U_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)\n');
    fprintf(fid,'    U_Mode1_RP_ZpYmXm = U_SubField.values[0].data\n');
    fprintf(fid,'    if (abs(UR_Mode1_RP_ZpYmXm[2]) > 1.0e-4) and (abs(U_Mode1_RP_ZpYmXm[0]) < 1.0e-4) and (abs(U_Mode1_RP_ZpYmXm[1]) < 1.0e-4):\n');
    fprintf(fid,'        coilable[iFrame_step-1] = 1\n');
    fprintf(fid,'    else:\n');
    fprintf(fid,'        coilable[iFrame_step-1] = 0\n');
    fprintf(fid,'\n');
    
    
%     fprintf(fid,'#\n');
%     fprintf(fid,'P_crit=Pcrit_perturb[1]\n');
    fprintf(fid,'#\n');
%     fprintf(fid,'UR_Field = MODELframe.fieldOutputs[''UR'']\n');
%     fprintf(fid,'UR_SubField = UR_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)\n');
%     fprintf(fid,'UR_Mode1_RP_ZpYmXm = UR_SubField.values[0].data\n');
%     fprintf(fid,'U_Field = MODELframe.fieldOutputs[''U'']\n');
%     fprintf(fid,'U_SubField = U_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)\n');
%     fprintf(fid,'U_Mode1_RP_ZpYmXm = U_SubField.values[0].data\n');
%     fprintf(fid,'if (abs(UR_Mode1_RP_ZpYmXm[2]) > 1.0e-4) and (abs(U_Mode1_RP_ZpYmXm[0]) < 1.0e-4) and (abs(U_Mode1_RP_ZpYmXm[1]) < 1.0e-4):\n');
%     fprintf(fid,'    coilable = 1\n');
%     fprintf(fid,'else:\n');
%     fprintf(fid,'    coilable = 0\n');
%     fprintf(fid,'\n');
    %%%%%%%%%%%%%%%%%%%
end

%% If it is a LINEAR Buckling analysis:
if UserSettings.analysis_option == 1 % If just doing LINEAR buckling, get max displacement from that file too
    % Preallocate variables at nodes
    postproc_variables_nodes = {'UR'};
%     postproc_variables_nodes = {'LE'};
    for ivar = 1:length(postproc_variables_nodes)
        fprintf(fid,'%s_Field = MODELframe.fieldOutputs[''%s'']\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'%s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_nSet, position=NODAL)\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
%         fprintf(fid,'%s_SubField = %s_Field.getSubset(region=RP_Zplus_nSet, position=NODAL)\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'#\n');
        fprintf(fid,'if isinstance(%s_SubField.values[0].data,float):\n',postproc_variables_nodes{ivar});
        fprintf(fid,'    # Then variable is a scalar\n');
        fprintf(fid,'    max_%s = numpy.ones(( numFrames ))*(-1e20)\n',postproc_variables_nodes{ivar});
        fprintf(fid,'else:\n');
        fprintf(fid,'    # Variable is an array\n');
        fprintf(fid,'    max_%s = numpy.ones(( numFrames,len(%s_SubField.values[0].data) ))*(-1e20)\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'\n');
    end
    
    % Extract values of variables
    fprintf(fid,'for iFrame_step in range(numFrames):\n');
    fprintf(fid,'    # Read frame\n');
    fprintf(fid,'    MODELframe = MODELodb.steps[mySteps.keys()[0]].frames[iFrame_step]\n');
%     fprintf(fid,'        eigenValue = [float(x) for x in re.findall(match_number, MODELframe.description)]\n');
%     fprintf(fid,'        #eigenValue[1] is the eigenValue for Mode = eigenValue[0]\n');
%     fprintf(fid,'        MOMcrit_perturb[iFrame_step,0]=eigenValue[0]\n');
%     fprintf(fid,'        MOMcrit_perturb[iFrame_step,1]=eigenValue[1]\n');
%     fprintf(fid,'    #\n');
    for ivar = 1:length(postproc_variables_nodes)
        fprintf(fid,'    # Variable: %s\n',postproc_variables_nodes{ivar});
        fprintf(fid,'    %s_Field = MODELframe.fieldOutputs[''%s'']\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'    %s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_nSet, position=NODAL)\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
%         fprintf(fid,'    %s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_nSet, position=NODAL)\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
%         fprintf(fid,'    %s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_nSet, position=NODAL)\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'    #\n');
        fprintf(fid,'    if isinstance(%s_SubField.values[0].data,float):\n',postproc_variables_nodes{ivar});
        fprintf(fid,'        # Then variable is a scalar:\n');
        fprintf(fid,'        for strainValue in %s_SubField.values:\n',postproc_variables_nodes{ivar});
        fprintf(fid,'            if (strainValue.data > max_%s[iFrame_step]):\n',postproc_variables_nodes{ivar});
        fprintf(fid,'                max_%s[iFrame_step] = abs(strainValue.data[0])\n',postproc_variables_nodes{ivar});
        fprintf(fid,'            #\n');
        fprintf(fid,'    else:\n');
        fprintf(fid,'        # Variable is an array:\n');
        fprintf(fid,'        for strainValue in %s_SubField.values:\n',postproc_variables_nodes{ivar});
        fprintf(fid,'            for j in range(0, len(%s_SubField.values[0].data)):\n',postproc_variables_nodes{ivar});
        fprintf(fid,'                if (strainValue.data[j] > max_%s[iFrame_step,j]):\n',postproc_variables_nodes{ivar});
        fprintf(fid,'                    max_%s[iFrame_step,j] = abs(strainValue.data[j])\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'                #\n');
        fprintf(fid,'    #\n');
        fprintf(fid,'    # Finished saving this variable!\n');
    end
    
    % Now compute the maximum displacement (for later scaling of the buckling modes):
    fprintf(fid,'    # Now compute the MAXIMUM of the 3 components of U for this frame:\n');
    fprintf(fid,'    maxDisp[iFrame_step] = max(max_%s[iFrame_step,:]) \n',postproc_variables_nodes{1});
    fprintf(fid,'#\n');
    fprintf(fid,'# Save variables of interest to a database\n');
    % Now take care of the variables that are always outputted, independent of
    % user choice:
%     permanent_vars = '''MOM_crit'':MOM_crit';
    permanent_vars = ['''','P_',sign_string,num2str(round(abs(load_dof))),'_crit','''',':P_crit',', ',...
        '''','maxDisp_',sign_string,num2str(round(abs(load_dof))),'''',':maxDisp',', ',...
        '''','coilable','''',':coilable'];
    %
    %
    %
%% If it is a NONLINEAR Buckling analysis:
elseif UserSettings.analysis_option == 2 
    fprintf(fid,'# Name of the file job\n');
    fprintf(fid,'jobname=''DoE%i_implicit_static''\n',jDoE);
    fprintf(fid,'#\n');
    %     fprintf(fid,'# Define the name of the matrix part\n');
    %     fprintf(fid,'matrix_part=''FinalRVE-1.Matrix_only''\n');
    %     fprintf(fid,'#\n');
    %     fprintf(fid,'# Define the name of the Particles part\n');
    %     fprintf(fid,'particle_part=''FinalRVE-1.Particle''\n');
    %
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
    fprintf(fid,'RP_Zplus_nSet = MODELodb.rootAssembly.nodeSets[''RP_ZPLUS'']\n');
    fprintf(fid,'RP_Zminus_nSet = MODELodb.rootAssembly.nodeSets[''RP_ZMINUS'']\n');
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
    % Preallocate variables at nodes
%     postproc_variables = {};
    postproc_variables = {'UR', 'CM'};
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
    postproc_variables_int = {};
%     postproc_variables_int = {'LE'};
    for ivar = 1:length(postproc_variables_int)
        fprintf(fid,'%s_Field = MODELframe.fieldOutputs[''%s'']\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        fprintf(fid,'%s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_elSet, position=INTEGRATION_POINT)\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        fprintf(fid,'%s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_elSet, position=INTEGRATION_POINT)\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        fprintf(fid,'#\n');
        fprintf(fid,'if isinstance(%s_SubField.values[0].data,float):\n',postproc_variables_int{ivar});
        fprintf(fid,'    # Then variable is a scalar\n');
        fprintf(fid,'    max_%s = numpy.ones(( totalNumFrames ))*(-1e20)\n',postproc_variables_int{ivar});
        fprintf(fid,'else:\n');
        fprintf(fid,'    # Variable is an array\n');
        fprintf(fid,'    max_%s = numpy.ones(( totalNumFrames,len(%s_SubField.values[0].data) ))*(-1e20)\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        fprintf(fid,'\n');
    end
    %
    % If you want to get the Moment-Angle curve from the implicit static simulations:  
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
    fprintf(fid,'ANG = numpy.zeros(( totalNumFrames ))\n');
    fprintf(fid,'MOM = numpy.zeros(( totalNumFrames ))\n');
    fprintf(fid,'maxLE11 = numpy.zeros(( totalNumFrames ))\n');
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
    fprintf(fid,'    # Now compute the critical buckling moment and angle:\n');
    fprintf(fid,'    ANG[iFrame_step] = %s_RP_Zplus[iFrame_step,%i] - %s_RP_Zminus[iFrame_step,%i] \n',postproc_variables{1},round(abs(load_dof))-4,postproc_variables{1},round(abs(load_dof))-4);
    fprintf(fid,'    MOM[iFrame_step] = %s_RP_Zplus[iFrame_step,%i]/2 \n',postproc_variables{2},round(abs(load_dof))-4);
    fprintf(fid,'    # Now compute other local quantities:\n');
    %
    % Extract values of variables
    for ivar = 1:length(postproc_variables_int)
        fprintf(fid,'    # Variable: %s\n',postproc_variables_int{ivar});
        fprintf(fid,'    %s_Field = MODELframe.fieldOutputs[''%s'']\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        fprintf(fid,'    %s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_elSet, position=INTEGRATION_POINT)\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        fprintf(fid,'    %s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_elSet, position=INTEGRATION_POINT)\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        fprintf(fid,'    #\n');
%         fprintf(fid,'    max = max(%s_SubField.values)\n',postproc_variables_int{ivar});
        fprintf(fid,'    if isinstance(%s_SubField.values[0].data,float):\n',postproc_variables_int{ivar});
        fprintf(fid,'        # Then variable is a scalar:\n');
        fprintf(fid,'        for strainValue in %s_SubField.values:\n',postproc_variables_int{ivar});
        fprintf(fid,'            if (strainValue.data > max_%s[iFrame]):\n',postproc_variables_int{ivar});
        fprintf(fid,'                max_%s[iFrame_step] = strainValue.data[0]\n',postproc_variables_int{ivar});
        fprintf(fid,'            #\n');
        fprintf(fid,'    else:\n');
        fprintf(fid,'        # Variable is an array:\n');
        fprintf(fid,'        for strainValue in %s_SubField.values:\n',postproc_variables_int{ivar});
        fprintf(fid,'                for j in range(0, len(%s_SubField.values[0].data)):\n',postproc_variables_int{ivar});
        fprintf(fid,'                    if (strainValue.data[j] > max_%s[iFrame_step,j]):\n',postproc_variables_int{ivar});
        fprintf(fid,'                        max_%s[iFrame_step,j] = strainValue.data[j]\n',postproc_variables_int{ivar},postproc_variables_int{ivar});
        fprintf(fid,'                    #\n');
        fprintf(fid,'    #\n');
        fprintf(fid,'    # Finished saving this variable!\n');
    end
    % Now compute the maximum log strain:
%     fprintf(fid,'    # Now compute the MAXIMUM log strain 11 (LE11):\n');
%     fprintf(fid,'    maxLE11[iFrame_step] = max_%s[iFrame_step,0] \n',postproc_variables_int{1});
    %
    % Save variables in database
    fprintf(fid,'\n');
    fprintf(fid,'#\n');
    %
    % NOW find the critical buckling moment by running multiple *BUCKLE analyses
%     fprintf(fid,'import re\n');
%     fprintf(fid,'match_number = re.compile(''-?\\ *[0-9]+\\.?[0-9]*(?:[Ee]\\ *[-+]?\\ *[0-9]+)?'')\n');
    fprintf(fid,'for iFrame_step in xrange(numFrames-1,0,-1):\n');
    fprintf(fid,'    MODELframe = MODELodb.steps[stepKey].frames[iFrame_step]\n');
    fprintf(fid,'    #\n');
    fprintf(fid,'    os.system(''rm DoE%i_nonlinear_buckle.*'')\n',jDoE);
    fprintf(fid,'    with open(''DoE%i_nonlinear_buckle.inp'',''w'') as File:\n',jDoE);
    fprintf(fid,'        File.write(''*Heading\\n'')\n');
    fprintf(fid,'        File.write(''** Job name: DoE%i_nonlinear_buckle Model name: DoE%i_nonlinear_buckle\\n'')\n',jDoE,jDoE);
    fprintf(fid,'        File.write(''*Preprint, echo=NO, model=NO, history=NO, contact=NO\\n'')\n');
    fprintf(fid,'        File.write(''*Restart, read, step=1, inc=''+str(iFrame_step)+'', END STEP\\n'')\n');
    fprintf(fid,'        File.write(''** ----------------------------------------------------------------\\n'')\n');
    fprintf(fid,'        File.write(''** STEP: Step-2\\n'')\n');
    fprintf(fid,'        File.write(''** \\n'')\n');
    fprintf(fid,'        File.write(''*Step\\n'')\n');
    fprintf(fid,'        File.write(''*Buckle, eigensolver=lanczos\\n'')\n');
    fprintf(fid,'        File.write(''20, , , , \\n'')\n');
    fprintf(fid,'        File.write(''** \\n'')\n');
    fprintf(fid,'        File.write(''** LOADS\\n'')\n');
    fprintf(fid,'        File.write(''** \\n'')\n');
    fprintf(fid,'        File.write(''** Name: Applied_Moment   Type: Moment\\n'')\n');
    fprintf(fid,'        File.write(''*Cload\\n'')\n');
    fprintf(fid,'        File.write(''RP_Zplus, %i, %6.2f\\n'')\n',round(abs(load_dof)),(sign(load_dof)*2.0));
    fprintf(fid,'        File.write(''** \\n'')\n');
    fprintf(fid,'        File.write(''*Node File\\n'')\n');
    fprintf(fid,'        File.write(''U \\n'')\n');
    fprintf(fid,'        File.write(''** \\n'')\n');
    fprintf(fid,'        File.write(''*EL PRINT,FREQUENCY=1\\n'')\n');
    fprintf(fid,'        File.write(''*NODE PRINT,FREQUENCY=1\\n'')\n');
    fprintf(fid,'        File.write(''*MODAL FILE\\n'')\n');
    fprintf(fid,'        File.write(''*OUTPUT,FIELD,VAR=PRESELECT\\n'')\n');
    fprintf(fid,'        File.write(''*OUTPUT,HISTORY,FREQUENCY=1\\n'')\n');
    fprintf(fid,'        File.write(''*MODAL OUTPUT\\n'')\n');
    fprintf(fid,'        File.write(''*End Step\\n'')\n');
%     fprintf(fid,'    File.close()\n');
    %
    % Run the Buckle Analysis to find critical buckling load:
    fprintf(fid,'    #\n');
    % fprintf(fid,'a = mdb.models[''Model-1''].rootAssembly\n');
    fprintf(fid,'    # Create job, run it and wait for completion\n');
    fprintf(fid,'    os.system(''abaqus j=DoE%i_nonlinear_buckle oldj=DoE%i_implicit_static interactive'')\n',jDoE,jDoE);
    % fprintf(fid,'    inputFile_string = ''%s''+''/''+''DoE%i_nonlinear_buckle.inp''\n',jDoE_dir,jDoE);
    % fprintf(fid,'    job=mdb.JobFromInputFile(name=''DoE%i_nonlinear_buckle'', inputFileName=inputFile_string, \n',jDoE);
    % fprintf(fid,'        type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, \n');
    % fprintf(fid,'        memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, \n');
    % fprintf(fid,'        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, userSubroutine='''', \n');
    % fprintf(fid,'        scratch='''', resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN, \n');
    % fprintf(fid,'        numDomains=1, activateLoadBalancing=False, multiprocessingMode=DEFAULT, \n');
    % fprintf(fid,'        numCpus=1)\n');
    % fprintf(fid,'    #\n');
    % fprintf(fid,'    job.submit()\n');
    % fprintf(fid,'    job.waitForCompletion()\n');
    %
    % TRY to open odb file and determine critical load
    fprintf(fid,'    try:\n');
    fprintf(fid,'        # Try to open this odb file\n');
    fprintf(fid,'        #Import Abaqus odb for the Buckle analysis\n');
    fprintf(fid,'        BUCKLE_MODELodb=openOdb(''DoE%i_nonlinear_buckle''+''.odb'')\n',jDoE);
    fprintf(fid,'        #\n');
    fprintf(fid,'        # Determine the number of steps in the output database.\n');
    fprintf(fid,'        BUCKLE_mySteps = BUCKLE_MODELodb.steps\n');
    fprintf(fid,'        BUCKLE_numSteps = len(BUCKLE_mySteps)\n');
    fprintf(fid,'        BUCKLE_stepKey = BUCKLE_mySteps.keys()[0]\n');
    fprintf(fid,'        BUCKLE_step = BUCKLE_mySteps[BUCKLE_stepKey]\n');
    fprintf(fid,'        BUCKLE_numFrames = len(BUCKLE_step.frames)\n');
    fprintf(fid,'        #\n');
    fprintf(fid,'        # Read critical buckling load\n');
    fprintf(fid,'        BUCKLE_MODELframe = BUCKLE_MODELodb.steps[BUCKLE_mySteps.keys()[0]].frames[1]\n');
    fprintf(fid,'        eigenValue = [float(x) for x in re.findall(match_number, BUCKLE_MODELframe.description)]\n');
    fprintf(fid,'        #eigenValue[1] is the eigenValue for Mode = eigenValue[0]\n');
    fprintf(fid,'        MOMcrit_perturb=eigenValue\n');
    fprintf(fid,'        #\n');
%     fprintf(fid,'        MOMcrit_perturb=numpy.zeros(( numFrames,2 ))\n');
%     fprintf(fid,'        for iFrame_step in range(numFrames):\n');
%     fprintf(fid,'            # Read critical buckling load\n');
%     fprintf(fid,'            MODELframe = MODELodb.steps[mySteps.keys()[0]].frames[iFrame_step]\n');
%     fprintf(fid,'            eigenValue = [float(x) for x in re.findall(match_number, MODELframe.description)]\n');
%     fprintf(fid,'            #eigenValue[1] is the eigenValue for Mode = eigenValue[0]\n');
%     fprintf(fid,'            MOMcrit_perturb[iFrame_step,0]=eigenValue[0]\n');
%     fprintf(fid,'            MOMcrit_perturb[iFrame_step,1]=eigenValue[1]\n');
%     fprintf(fid,'        #\n');
    fprintf(fid,'        MOM_crit=abs(MOM[iFrame_step])+MOMcrit_perturb[1]\n');
    fprintf(fid,'        #\n');
    fprintf(fid,'        break # We found the buckling load!\n');
    fprintf(fid,'    except Exception, e:\n');
    fprintf(fid,'        print >> sys.stderr, ''does not exist''\n');
    fprintf(fid,'        print >> sys.stderr, ''Exception: %%s'' %% str(e)\n');
    fprintf(fid,'        if iFrame_step == 0:\n');
    fprintf(fid,'            sys.exit(1) # Exit the code because there is nothing left to do!\n');
    fprintf(fid,'    #\n');
    fprintf(fid,'\n');
    %
    % Now compute the max displacements (for later scaling of imperfection
    % modes):
    % Preallocate variables at nodes
    postproc_variables_nodes = {'U'};
    %
    fprintf(fid,'maxDisp = numpy.zeros(( BUCKLE_numFrames ))\n');
%     fprintf(fid,'RP_Zplus_nSet = BUCKLE_MODELodb.rootAssembly.nodeSets[''RP_ZPLUS'']\n');
%     fprintf(fid,'RP_Zminus_nSet = BUCKLE_MODELodb.rootAssembly.nodeSets[''RP_ZMINUS'']\n');
    fprintf(fid,'entireSTRUCTURE_nSet = BUCKLE_MODELodb.rootAssembly.nodeSets['' ALL NODES'']\n');
    fprintf(fid,'entireSTRUCTURE_elSet = BUCKLE_MODELodb.rootAssembly.elementSets['' ALL ELEMENTS'']\n');
    fprintf(fid,'#\n');
    %
    for ivar = 1:length(postproc_variables_nodes)
        fprintf(fid,'%s_Field = BUCKLE_MODELframe.fieldOutputs[''%s'']\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'%s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_nSet, position=NODAL)\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'%s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_nSet, position=NODAL)\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'#\n');
        fprintf(fid,'if isinstance(%s_SubField.values[0].data,float):\n',postproc_variables_nodes{ivar});
        fprintf(fid,'    # Then variable is a scalar\n');
        fprintf(fid,'    max_%s = numpy.ones(( BUCKLE_numFrames ))*(-1e20)\n',postproc_variables_nodes{ivar});
        fprintf(fid,'else:\n');
        fprintf(fid,'    # Variable is an array\n');
        fprintf(fid,'    max_%s = numpy.ones(( BUCKLE_numFrames,len(%s_SubField.values[0].data) ))*(-1e20)\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'\n');
    end
    
    % Extract values of variables
    %
    fprintf(fid,'for iFrame_step in range(BUCKLE_numFrames):\n');
    fprintf(fid,'    # Read frame\n');
    fprintf(fid,'    BUCKLE_MODELframe = BUCKLE_MODELodb.steps[BUCKLE_mySteps.keys()[0]].frames[iFrame_step]\n');
%     fprintf(fid,'        eigenValue = [float(x) for x in re.findall(match_number, MODELframe.description)]\n');
%     fprintf(fid,'        #eigenValue[1] is the eigenValue for Mode = eigenValue[0]\n');
%     fprintf(fid,'        MOMcrit_perturb[iFrame_step,0]=eigenValue[0]\n');
%     fprintf(fid,'        MOMcrit_perturb[iFrame_step,1]=eigenValue[1]\n');
%     fprintf(fid,'    #\n');
    for ivar = 1:length(postproc_variables_nodes)
        fprintf(fid,'    # Variable: %s\n',postproc_variables_nodes{ivar});
        fprintf(fid,'    %s_Field = BUCKLE_MODELframe.fieldOutputs[''%s'']\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'    %s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_nSet, position=NODAL)\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'    %s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_nSet, position=NODAL)\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'    #\n');
        fprintf(fid,'    if isinstance(%s_SubField.values[0].data,float):\n',postproc_variables_nodes{ivar});
        fprintf(fid,'        # Then variable is a scalar:\n');
        fprintf(fid,'        for strainValue in %s_SubField.values:\n',postproc_variables_nodes{ivar});
        fprintf(fid,'            if (strainValue.data > max_%s[iFrame_step]):\n',postproc_variables_nodes{ivar});
        fprintf(fid,'                max_%s[iFrame_step] = abs(strainValue.data[0])\n',postproc_variables_nodes{ivar});
        fprintf(fid,'            #\n');
        fprintf(fid,'    else:\n');
        fprintf(fid,'        # Variable is an array:\n');
        fprintf(fid,'        for strainValue in %s_SubField.values:\n',postproc_variables_nodes{ivar});
        fprintf(fid,'            for j in range(0, len(%s_SubField.values[0].data)):\n',postproc_variables_nodes{ivar});
        fprintf(fid,'                if (strainValue.data[j] > max_%s[iFrame_step,j]):\n',postproc_variables_nodes{ivar});
        fprintf(fid,'                    max_%s[iFrame_step,j] = abs(strainValue.data[j])\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'                #\n');
        fprintf(fid,'    #\n');
        fprintf(fid,'    # Finished saving this variable!\n');
    end
    %
    % Now compute the maximum displacement (for later scaling of the buckling modes):
    fprintf(fid,'    # Now compute the MAXIMUM of the 3 components of U for this frame:\n');
    fprintf(fid,'    maxDisp[iFrame_step] = max(max_%s[iFrame_step,:]) \n',postproc_variables_nodes{1});
    fprintf(fid,'#\n');
    %
    %
    fprintf(fid,'# Save variables of interest to a database\n');
    %
    % If you just want to output the critical load:
%     permanent_vars = ['''','MOM_',sign_string,num2str(round(abs(load_dof))),'_crit','''',':MOM_crit'];
    % If you want to output critical load and the load-displacement curve:
    permanent_vars = ['''','MOM_',sign_string,num2str(round(abs(load_dof))),'_crit','''',':MOM_crit',', ',...
        '''','maxDisp_',sign_string,num2str(round(abs(load_dof))),'''',':maxDisp'];%,', ',...
%         '''','ANG_',sign_string,num2str(round(abs(load_dof))),'''',':ANG',', ',...
%         '''','MOM_',sign_string,num2str(round(abs(load_dof))),'''',':MOM'];%,', ',...
%         '''','maxLE11_',sign_string,num2str(round(abs(load_dof))),'''',':maxLE11'];
%     permanent_vars = '''MOM_crit'':MOM_crit, ''ANG'':ANG, ''MOM'':MOM';
%     permanent_vars = ['''','MOM_',sign_string,num2str(round(abs(load_dof))),'_crit','''',':MOM_crit, ','''','ANG_',sign_string,num2str(round(abs(load_dof))),'''',':ANG, ','''','MOM_',sign_string,num2str(round(abs(load_dof))),'''',':MOM'];
    %
    %
    %
%% If it is a FREQUENCY analysis:
elseif UserSettings.analysis_option == 3
    % TRY to open odb file and determine critical load
%     fprintf(fid,'import re\n');
%     fprintf(fid,'match_number = re.compile(''-?\\ *[0-9]+\\.?[0-9]*(?:[Ee]\\ *[-+]?\\ *[0-9]+)?'')\n');
    fprintf(fid,'try:\n');
    fprintf(fid,'    # Try to open this odb file\n');
    fprintf(fid,'    #Import Abaqus odb for the frequency analysis\n');
    fprintf(fid,'    MODELodb=openOdb(''DoE%i_frequency''+''.odb'')\n',jDoE);
    fprintf(fid,'    #\n');
    
    %%%%%%%%%%%%%%%%%%%
    fprintf(fid,'except Exception, e:\n');
    fprintf(fid,'    print >> sys.stderr, ''does not exist''\n');
    fprintf(fid,'    print >> sys.stderr, ''Exception: %%s'' %% str(e)\n');
    fprintf(fid,'    sys.exit(1) # Exit the code because there is nothing left to do!\n');
    fprintf(fid,'#\n');
    fprintf(fid,'\n');
    
    
    fprintf(fid,'# Determine the number of steps in the output database.\n');
    fprintf(fid,'mySteps = MODELodb.steps\n');
    fprintf(fid,'numSteps = len(mySteps)\n');
    fprintf(fid,'stepKey = mySteps.keys()[0]\n');
    fprintf(fid,'step = mySteps[stepKey]\n');
    fprintf(fid,'numFrames = len(step.frames)\n');
    fprintf(fid,'#\n');
    fprintf(fid,'maxDisp = numpy.zeros(( numFrames ))\n');
    fprintf(fid,'RP_Zplus_nSet = MODELodb.rootAssembly.nodeSets[''RP_ZPLUS'']\n');
    fprintf(fid,'RP_Zminus_nSet = MODELodb.rootAssembly.nodeSets[''RP_ZMINUS'']\n');
    fprintf(fid,'entireSTRUCTURE_nSet = MODELodb.rootAssembly.nodeSets['' ALL NODES'']\n');
    fprintf(fid,'entireSTRUCTURE_elSet = MODELodb.rootAssembly.elementSets['' ALL ELEMENTS'']\n');
    fprintf(fid,'#\n');
    fprintf(fid,'# Read critical buckling load\n');
    fprintf(fid,'MODELframe = MODELodb.steps[mySteps.keys()[0]].frames[1]\n');
    fprintf(fid,'eigenFreq = [float(x) for x in re.findall(match_number, MODELframe.description)]\n');
    fprintf(fid,'#eigenFreq[1] is the eigen Frequency for Mode = eigenFreq[0]\n');
    fprintf(fid,'NatFreq_perturb=eigenFreq\n');
    fprintf(fid,'#\n');
%     fprintf(fid,'MOMcrit_perturb=numpy.zeros(( numFrames,2 ))\n');
%     fprintf(fid,'for iFrame_step in range(numFrames):\n');
%     fprintf(fid,'    # Read critical buckling load\n');
%     fprintf(fid,'    MODELframe = MODELodb.steps[mySteps.keys()[0]].frames[iFrame_step]\n');
%     fprintf(fid,'    eigenFreq = [float(x) for x in re.findall(match_number, MODELframe.description)]\n');
%     fprintf(fid,'    #eigenFreq[1] is the eigen Frequency for Mode = eigenFreq[0]\n');
%     fprintf(fid,'    NatFreq_perturb[iFrame_step,0]=eigenFreq[0]\n');
%     fprintf(fid,'    NatFreq_perturb[iFrame_step,1]=eigenFreq[1]\n');
%     fprintf(fid,'#\n');
    fprintf(fid,'NatFreq=NatFreq_perturb[2]\n');
    fprintf(fid,'#\n');
    %%%%%%%%%%%%%%%%%%%
    
    % Preallocate variables at nodes
    postproc_variables_nodes = {'U'};
    for ivar = 1:length(postproc_variables_nodes)
        fprintf(fid,'%s_Field = MODELframe.fieldOutputs[''%s'']\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'%s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_nSet, position=NODAL)\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'%s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_nSet, position=NODAL)\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'#\n');
        fprintf(fid,'if isinstance(%s_SubField.values[0].data,float):\n',postproc_variables_nodes{ivar});
        fprintf(fid,'    # Then variable is a scalar\n');
        fprintf(fid,'    max_%s = numpy.ones(( numFrames ))*(-1e20)\n',postproc_variables_nodes{ivar});
        fprintf(fid,'else:\n');
        fprintf(fid,'    # Variable is an array\n');
        fprintf(fid,'    max_%s = numpy.ones(( numFrames,len(%s_SubField.values[0].data) ))*(-1e20)\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'\n');
    end
    
    % Extract values of variables
    
    
    fprintf(fid,'for iFrame_step in range(numFrames):\n');
    fprintf(fid,'    # Read frame\n');
    fprintf(fid,'    MODELframe = MODELodb.steps[mySteps.keys()[0]].frames[iFrame_step]\n');
%     fprintf(fid,'        eigenValue = [float(x) for x in re.findall(match_number, MODELframe.description)]\n');
%     fprintf(fid,'        #eigenValue[1] is the eigenValue for Mode = eigenValue[0]\n');
%     fprintf(fid,'        MOMcrit_perturb[iFrame_step,0]=eigenValue[0]\n');
%     fprintf(fid,'        MOMcrit_perturb[iFrame_step,1]=eigenValue[1]\n');
%     fprintf(fid,'    #\n');
    for ivar = 1:length(postproc_variables_nodes)
        fprintf(fid,'    # Variable: %s\n',postproc_variables_nodes{ivar});
        fprintf(fid,'    %s_Field = MODELframe.fieldOutputs[''%s'']\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'    %s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_nSet, position=NODAL)\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'    %s_SubField = %s_Field.getSubset(region=entireSTRUCTURE_nSet, position=NODAL)\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'    #\n');
        fprintf(fid,'    if isinstance(%s_SubField.values[0].data,float):\n',postproc_variables_nodes{ivar});
        fprintf(fid,'        # Then variable is a scalar:\n');
        fprintf(fid,'        for strainValue in %s_SubField.values:\n',postproc_variables_nodes{ivar});
        fprintf(fid,'            if (strainValue.data > max_%s[iFrame_step]):\n',postproc_variables_nodes{ivar});
        fprintf(fid,'                max_%s[iFrame_step] = abs(strainValue.data[0])\n',postproc_variables_nodes{ivar});
        fprintf(fid,'            #\n');
        fprintf(fid,'    else:\n');
        fprintf(fid,'        # Variable is an array:\n');
        fprintf(fid,'        for strainValue in %s_SubField.values:\n',postproc_variables_nodes{ivar});
        fprintf(fid,'            for j in range(0, len(%s_SubField.values[0].data)):\n',postproc_variables_nodes{ivar});
        fprintf(fid,'                if (strainValue.data[j] > max_%s[iFrame_step,j]):\n',postproc_variables_nodes{ivar});
        fprintf(fid,'                    max_%s[iFrame_step,j] = abs(strainValue.data[j])\n',postproc_variables_nodes{ivar},postproc_variables_nodes{ivar});
        fprintf(fid,'                #\n');
        fprintf(fid,'    #\n');
        fprintf(fid,'    # Finished saving this variable!\n');
    end
    
    % Now compute the maximum displacement (for later scaling of the buckling modes):
    fprintf(fid,'    # Now compute the MAXIMUM of the 3 components of U for this frame:\n');
    fprintf(fid,'    maxDisp[iFrame_step] = max(max_%s[iFrame_step,:]) \n',postproc_variables_nodes{1});
    fprintf(fid,'#\n');
    fprintf(fid,'# Save variables of interest to a database\n');
    % Now take care of the variables that are always outputted, independent of
    % user choice:
%     permanent_vars = '''MOM_crit'':MOM_crit';
    permanent_vars = ['''','NatFreq_',sign_string,num2str(round(abs(load_dof))),'''',':NatFreq',', ',...
        '''','MOM_',sign_string,num2str(round(abs(load_dof))),'_crit','''',':MOM_crit',', ',...
        '''','maxDisp_',sign_string,num2str(round(abs(load_dof))),'''',':maxDisp'];
    %
    %
    %
end
%
% Save STRUCTURE_variables:
fprintf(fid,'STRUCTURE_variables = {%s}\n',permanent_vars);
fprintf(fid,'#\n');
fprintf(fid,'stringDoE = ''DoE''+str(%i)\n',jDoE);
fprintf(fid,'stringImperfection = ''Imperfection''+str(%i)\n',kImperfection);
fprintf(fid,'stringInput = ''Input''+str(%i)\n',iInput);
%     fprintf(fid,'stringSample = ''Sample''+str(%i)\n',kSample);
%     fprintf(fid,'stringMATs = ''MATs''+str(%i)\n',iMATs);
%     fprintf(fid,'stringBC = ''BC''+str(%i)\n',iBC);
%fprintf(fid,'STRUCTURES_data_update = {stringInput : {stringDoE : {stringImperfection : STRUCTURE_variables } } }\n');
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
fprintf(fid,'# End of file.\n');
%
%
fclose(fid); % Close file
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Execute Python Script in ABAQUS CAE                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
disp(' ');
disp(['Started Abaqus odb post-processing for DoE',num2str(jDoE),' Imperfection',num2str(kImperfection)]);
disp(' ');
unix(strcat(UserSettings.abaqus_path,' cae noGUI=',postproc_dir,'/script_write_postproc_CPU',num2str(CPUid),'.py'));
disp(' ');
disp(['STRUCTURES_postprocessing_variables.p file UPDATED for DoE',num2str(jDoE),' Imperfection',num2str(kImperfection)]);
% disp('Elapsed Time [min]: ');
% disp(toc/60);
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of Function                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
