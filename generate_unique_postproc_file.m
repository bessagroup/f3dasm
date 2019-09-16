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
function generate_unique_postproc_file(UserSettings,postproc_dir)
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
% global UserSettings.load_dof;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create Python File for Post-processing Abaqus simulations               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% disp(' ');
% disp('Started creation of Python file for postprocessing');
%
%
matlab_cpus = UserSettings.matlab_cpus;
%
static_scriptfile_name = strcat(postproc_dir,'/script_merge_postproc_files.py');
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
% Select work directory:
% fprintf(fid,'os.chdir(r''%s'')\n',jDoE_dir);
%
fprintf(fid,'# Set directory where you want the post-processing file to be generated\n');
fprintf(fid,'postproc_dir=''%s''\n',postproc_dir);
fprintf(fid,'#\n');

fprintf(fid,'file_postproc_path = postproc_dir+''/''+''STRUCTURES_postprocessing_variables.p''\n');
fprintf(fid,'# Flag saying that the MERGED (unique) post-processing file exists:\n');
fprintf(fid,'try:\n');
fprintf(fid,'    # Try to load the previous (unique) post-processing file with all information:\n');
fprintf(fid,'    readfile_postproc = open(file_postproc_path, ''rb'')\n');
fprintf(fid,'    STRUCTURES_data = pickle.load(readfile_postproc)\n');
fprintf(fid,'    readfile_postproc.close()\n');
fprintf(fid,'    postproc_exists = 1\n');
fprintf(fid,'except Exception, e:\n');
fprintf(fid,'    postproc_exists = 0 # Flag saying that there is no post-processing file saved previously\n');
fprintf(fid,'\n');
fprintf(fid,'#\n');

for iCPU = 1:matlab_cpus
fprintf(fid,'file_postproc_path_CPU%s = postproc_dir+''/''+''STRUCTURES_postprocessing_variables_CPU%s.p''\n',num2str(iCPU),num2str(iCPU));
fprintf(fid,'# Flag saying post-processing file for this CPU exists:\n');
fprintf(fid,'try:\n');
fprintf(fid,'    # Try to load the previous post-processing file for this CPU with all information:\n');
fprintf(fid,'    readfile_postproc_CPU%s = open(file_postproc_path_CPU%s, ''rb'')\n',num2str(iCPU),num2str(iCPU));
fprintf(fid,'    STRUCTURES_data_CPU%s = pickle.load(readfile_postproc_CPU%s)\n',num2str(iCPU),num2str(iCPU));
fprintf(fid,'    readfile_postproc_CPU%s.close()\n',num2str(iCPU));
fprintf(fid,'    postproc_exists_CPU%s = 1\n',num2str(iCPU));
fprintf(fid,'except Exception, e:\n');
fprintf(fid,'    postproc_exists_CPU%s = 0 # Flag saying that there is no post-processing file saved previously\n',num2str(iCPU));
fprintf(fid,'\n');
fprintf(fid,'#\n');
%
fprintf(fid,'if (postproc_exists == 1) and (postproc_exists_CPU%s == 1):\n',num2str(iCPU));
fprintf(fid,'    STRUCTURES_data = dict_merge(STRUCTURES_data, STRUCTURES_data_CPU%s)\n',num2str(iCPU));
fprintf(fid,'elif (postproc_exists != 1) and (postproc_exists_CPU%s == 1):\n',num2str(iCPU));
fprintf(fid,'    STRUCTURES_data = STRUCTURES_data_CPU%s\n',num2str(iCPU));
fprintf(fid,'    postproc_exists = 1 # Flag saying that there is a post-processing file\n');
% fprintf(fid,'else:\n');
% fprintf(fid,'    # Nothing to do.\n');
fprintf(fid,'\n');
end
%
fprintf(fid,'# Save post-processing information to pickle file:\n');
fprintf(fid,'writefile_postproc = open(file_postproc_path, ''wb'')\n');
fprintf(fid,'pickle.dump(STRUCTURES_data, writefile_postproc)\n');
fprintf(fid,'writefile_postproc.close()\n');
fprintf(fid,'\n');
fprintf(fid,'#\n');
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Execute Python Script in ABAQUS CAE                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
disp(' ');
disp('Started Abaqus merge of post-processing files for every CPU used');
disp(' ');
unix(strcat(UserSettings.abaqus_path,' cae noGUI=',postproc_dir,'/script_merge_postproc_files.py'));
disp(' ');
disp('Finished updating the merged STRUCTURES_postprocessing_variables.p file');
% disp('Elapsed Time [min]: ');
% disp(toc/60);
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of Function                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
