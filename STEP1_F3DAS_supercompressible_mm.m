%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%            Matlab Routine for data-driven discovery of material         %
%               properties and constitutive behavior                      %
%                                                                         %
%                September 2019                                           %
%                                                                         %
%                Code developed by:                                       %
%                Miguel A. Bessa - M.A.Bessa@tudelft.nl                   %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
clc; close all; clear; tic; fclose('all');
% clc; close all; tic; fclose('all');
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%UserSettings.abaqus_path = '/opt/abaqus/Commands/abq6132';  % Specify path to ABAQUS command
UserSettings.abaqus_path = 'abaqus';  % Specify path to ABAQUS command
%
% % UserSettings.analysis_folder = 'DOE_Single_Astromat_1St_7ParamsFirst_Imperfect'; % Name of the folder where all the files for all the simulations will be written
UserSettings.analysis_folder = 'DOE_Single_Astromat_1St_7ParamsFirst_Imperfect_Coilable'; % Name of the folder where all the files for all the simulations will be written
% UserSettings.analysis_folder = 'AstroMat_5Params_PLA_manual'; % Name of the folder where all the files for all the simulations will be written
%
% Define the DoE:
UserSettings.DoE_file_name = 'DOE_TO_Classify_NEW'; % Name of the file with all the sample microstructures (without ".mat")
% UserSettings.DoE_file_name = 'DOE_Single_Astromat_1St_7ParamsFirst_Imperfect'; % Name of the file with all the sample microstructures (without ".mat")
UserSettings.DoE_size = 1.5E6; % Number of DoE points
% UserSettings.DoE_vars        = {'$D_1$' , '$\frac{P}{D_1}$' , '$\frac{d}{D_1}$', '$\frac{D_1-D_2}{D_1}$'}; % Input design variables names
UserSettings.DoE_vars        = {'$\frac{A}{D_1^2}$','$\frac{G}{E}$','$\frac{Ix}{D_1^4}$','$\frac{Iy}{D_1^4}$','$\frac{J}{D_1^4}$', '$\frac{P}{D_1}$' , '$\frac{D_1-D_2}{D_1}$'}; % Input design variables names
% UserSettings.DoE_vars        = {'P_D' , 'nStories', 'MastDiameter', 'ConeSlope' , 'Batten_d/Longeron_b' }; % Input design variables names
                                                %this is just for reference and to
                                                %confirm the dimension of the input
                                                %design space: dim = length(vars)
UserSettings.DoE_LowerBounds = [1.0e-5 , 0.335 , 1.1275e-11 , 1.1275e-11 , 1.353165e-11 , 0.25 , 0.0 ]; % (1 x dim vector) The lower bounds of each DoE variables.
UserSettings.DoE_UpperBounds = [4.096e-3 , 0.45 , 1.3981e-6 , 1.3981e-6, 6.76681e-6 , 1.5  , 0.8 ]; %(1 x dim vector) The upper bounds of each DoE variable.
% UserSettings.DoE_LowerBounds = [25  , 0.25 , 0.00625 , 0.0 ]; % (1 x dim vector) The lower bounds of each DoE variables.
% UserSettings.DoE_UpperBounds = [250 , 1.5  , 0.0625  , 0.8 ]; %(1 x dim vector) The upper bounds of each DoE variable.
% UserSettings.DoE_LowerBounds = [0.1 , 2 , 10.0  , 0.0 , 0.25 ]; % (1 x dim vector) The lower bounds of each DoE variable.
% UserSettings.DoE_UpperBounds = [1.0 , 20, 100.0 , 0.8 , 1.0  ]; %(1 x dim vector) The upper bounds of each DoE variable.
% UserSettings.DoE_LowerBounds = [10.0  , 2.0e-3 , 50.0e-6]; % (1 x dim vector) The lower bounds of each DoE variable.
% UserSettings.DoE_UpperBounds = [315.0 , 8.0e-3 , 500.0e-6]; %(1 x dim vector) The upper bounds of each DoE variable.
%
%
% Define the Fixed Inputs:
% UserSettings.Input_file_name = 'INPUT_D_PD_St_Slope'; % Name of the file with all the fixed INPUTS (without ".mat")
UserSettings.Input_file_name = 'DOE_Single_Astromat_1St_7ParamsFirst_Imperfect'; % Name of the file with all the fixed INPUTS (without ".mat")
UserSettings.Input_size = 1; % Number of Input points (sets of fixed values)
% UserSettings.Input_vars = {'VertexPolygon', 'power', ...
%     'Longeron_b', 'LongeronRatio_a_b', 'BattenRadius',...
%     'DiagonalRadius', 'Twist_angle', 'transition_length_ratio'}; % Fixed (Discrete) input variables
UserSettings.Input_vars = {'VertexPolygon', 'nStories', 'BattenDiamater/d', 'power', 'pinned_joints', ...
    'EModulus', 'D1', 'BattenFormat', 'BattenMaterial', ...
    'DiagonalRadius', 'DiagonalMaterial', 'Twist_angle', 'transition_length_ratio'}; % Fixed (Discrete) input variables
% UserSettings.Input_vars = {'VertexPolygon', 'nStories', 'BattenDiamater/d', 'power', 'LongeronRatio_a_b', 'pinned_joints', ...
%     'LongeronMaterial', 'BattenFormat', 'BattenMaterial', ...
%     'DiagonalRadius', 'DiagonalMaterial', 'Twist_angle', 'transition_length_ratio'}; % Fixed (Discrete) input variables
% EACH ROW OF THE FOLLOWING VARIABLE REPRESENTES A INPUT POINT (columns correspond to each variable)
UserSettings.Input_points = ...
    [ 3  , 1 , 1.0 , 1.0, 1  , ...
      1826.0, 100.0, 1   , 3  , ...
      0.0 , 3  , 0.0, 1.0; ... % end of this input point
      3  , 1 , 1.0 , 1.0, 1  , ...
      3500.0, 100.0, 1   , 3  , ...
      0.0 , 3  , 0.0, 1.0 ]; % end of this input point
% UserSettings.Input_points = ...
%     [ 10  , 1 , 0.5 , 1.0, 1.0, 0  , ...
%        3  , 1   , 3  , ...
%       0.0 , 3  , 0.0, 1.0 ; ... % end of first input point
%       10  , 1 , 1.0 , 1.0, 1.0, 0  , ...
%        3  , 1   , 3  , ...
%       0.0 , 3  , 0.0, 1.0 ]; % end of this input point
%       0.0 , 3  , 0.0, 1.0 ]; % Matrix of Input_size * dim(Input_vars)
%     [ 3 , 1.0 , 3.0 , 2.0 , 0.75 , 0.0 , 0.0 , 1.0 ]; % Matrix of Input_size * dim(Input_vars)
%     [ 3 , 1.0 , 6.0 , 4.0 , 1.5 , 0.0 , 0.0 , 1.0 ]; % Matrix of Input_size * dim(Input_vars)
%
%
% UserSettings.load_dof = 4; % Degree of freedom of loading condition
%
UserSettings.matlab_cpus = 4; % Number of "MATLAB workers" (CPUs)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Imperfection distribution settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Choose distribution (Gaussian, Lognormal, Gumbel, etc.):
UserSettings.Imperfection.distribution = 'Lognormal'; 

% Choose the values for the MOMENTS of the distribution:
% Example for Lognormal distribution: Moments = [mean, standard deviation]
% UserSettings.UQlab.Moments = [ 3.507e-02, 2.089e-02 ]; % Here the imperfection is mean and std of the angle of the top part of the structure (in radian)
UserSettings.Imperfection.Moments = [ 0.07, 0.021 ];
parameter2 = sqrt(log(((UserSettings.Imperfection.Moments(2)/UserSettings.Imperfection.Moments(1))^2)+1));
parameter1 = log(UserSettings.Imperfection.Moments(1)) - (parameter2^2)/2;

% ALTERNATIVELY to giving moments, you could give the direct PARAMETERS of
%the distribution. But UQlab already calculates these if you give the
%moments. I find the moments more intuitive quantities.
% UserSettings.UQlab.Parameters = [-3.5023 , 0.551];

% Choose the size of the sample to extract from the distribution (number of
%imperfections you want to include in the simulation):


%Generate the imperfection sample
% UserSettings.RIKS_imperfection = generate_imperfections(UserSettings.UQlab);
% The above vector contains the magnitude of the imperfections for successive buckling modes
% UserSettings.RIKS_imperfection = 0.01*ones(UserSettings.UQlab.samplesize,1);
UserSettings.RIKS_imperfection = random(UserSettings.Imperfection.distribution,parameter1,parameter2,UserSettings.DoE_size,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Option Parameters
% corrections = h5read('BuckleTop100noisy.h5','/BuckleTop100noisy');
% corrections = sort(transpose(corrections));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Essential options:
UserSettings.selectDoEs =1:12; % DoEs that will be analyzed.
                        %
UserSettings.selectInputs = 2;  % Input points that will be analyzed.
                          %
% UserSettings.selectImperfections = 1:UserSettings.UQlab.samplesize;
UserSettings.selectImperfections = 1:1; % Imperfections to sample for RIKS analysis
                             %
UserSettings.analysis_option = 1; % Type of analysis to run:
                             % -> 1-LINEAR Buckling analysis (FAST!)
                             % -> 2-NONLINEAR Buckling analysis (SLOWER
                             %      because it includes implicit static
                             %      analysis to determine buckling point
                             %      accurately)
                             % -> 3-FREQUENCY analysis (FAST!)
UserSettings.RIKS_option = 1; % Do Postbuckling analysis (SLOW!):
                         % -> 0-NO
                         % -> 1-YES
                         % NOTE: The RIKS analysis is based on the imperfections generated
                         %       from analysis_option = 1, 2 or 3. 
                         %       So, the RIKS results depend on the
                         %       previous choice being 1, 2, 3, etc.
                         %
UserSettings.run_simul_option = 1;      % Do you want to run the simulations? (NOTE: THIS ALSO CONTROLS THE RIKS OPTION!)
                              % -> 0-NO
                              % -> 1-YES, but running in local desktop
                              % -> 2-YES, but running in ARES cluster
                              %
UserSettings.postproc_option = 1;       % Do STRUCTURES postprocessing?
                             % -> 0-NO
                             % -> 1-YES
                             %
UserSettings.convert_p2mat_option = 1;  % Do you want to convert the python post-processing
                                  %file to a .mat file that MATLAB can read?
                                  % -> 0-NO
                                  % -> 1-YES
                                  %
UserSettings.delodb_option = 0; % Delete STRUCTURES odb file?
                           % -> 0-NO
                           % -> 1-YES
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definition of analysis to run
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if UserSettings.analysis_option == 2 % Implicit static
    UserSettings.analysis_parameters.init_time_inc = 0.05; % Initial time increment for simulation
    UserSettings.analysis_parameters.total_time = 1.0; % Total time of simulation
    UserSettings.analysis_parameters.min_time_inc = 1E-06; % Minimum time increment allowed
    UserSettings.analysis_parameters.max_time_inc = 0.05; % Maximum time increment allowed
    UserSettings.analysis_parameters.output_time_interval = 1.0; % Time interval to output to odb file
end
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation Inputs 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mesh size:
UserSettings.analysis_parameters.Mesh_size = 0.01; % Make sure that this is small enough! Should be below the minimum between the dimension
                  %of the smallest axis of the smallest particle
UserSettings.analysis_parameters.refine_factor = 1.25; % Parameter used to refine mesh (should be larger than 1)
UserSettings.analysis_parameters.niter = 2; % Maximum number of iterations to mesh the RVE
%
% If running the simulations in ARES cluster:
UserSettings.analysis_parameters.ncpus = 1; % Number of CPUs to use in ARES cluster OR in LOCAL DESKTOP
UserSettings.analysis_parameters.nnodes = 1; % Number of nodes in ARES cluster only
%
% What are the post-processing variables that you want?
UserSettings.postproc_variables_int = {}; % Variables that exist at INTEGRATION POINTS! (It can be an empty cell)
UserSettings.postproc_variables_whole = {}; % Variables that exist at the WHOLE ELEMENT! (It can be an empty cell)
UserSettings.postproc_variables_nodes = {'RF3','U3'}; % Variables that exist at the NODES! (It can be an empty cell)
%
UserSettings.postproc_compute_max = 1; % if UserSettings.postproc_compute_max = 1 it will calculate the maximum values of the requested variables
UserSettings.postproc_compute_min = 1; % if UserSettings.postproc_compute_min = 1 it will calculate the maximum values of the requested variables
UserSettings.postproc_compute_av = 0; % if UserSettings.postproc_compute_av = 1 it will calculate the maximum values of the requested variables
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create folders if they don't exist
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save folder where main function is located
pwd;
main_dir = pwd;
%
% Save path to microstructure generation files
DoE_dir = strcat(cd,'/1_DoEs');
if exist(DoE_dir,'dir') == 0, mkdir(DoE_dir); end
% Save path to Input files for material properties
input_dir = strcat(cd,'/2_Inputs');
if exist(input_dir,'dir') == 0, mkdir(input_dir); end
% Create a folder for this RVE, if it doesn't already exist:
analyses_dir = strcat(cd,'/3_Analyses/',UserSettings.analysis_folder);
% external_hardDrive = '/media/UserData/mabessa/Research/AstroMat_project';
% external_hardDrive = '/home/mabessa/Desktop/Abaqus_Analyses';
% analyses_dir = strcat(external_hardDrive,'/3_Analyses/',UserSettings.analysis_folder);
if exist(analyses_dir,'dir') == 0, mkdir(analyses_dir); end
% Create a folder for postprocessing, if it doesn't already exist:
postproc_dir = strcat(cd,'/4_Postprocessing/',UserSettings.analysis_folder);
if exist(postproc_dir,'dir') == 0, mkdir(postproc_dir); end
%
% Create file to indicate if there were ERRORS in generating the materials (ERROR_materials):
% Delete old ERROR_materials if it exists:
errorfile_name = strcat(analyses_dir,'/ERROR_FILE');
if exist(errorfile_name,'file') == 2, delete(errorfile_name); end
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create or LOAD the DoE file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the Design of Experiments (DoE):
if exist([DoE_dir,'/',UserSettings.DoE_file_name,'.mat'],'file') == 0
    %
    DoE_dim = length(UserSettings.DoE_vars);
    DoE_points = generate_DoE(DoE_dim, UserSettings.DoE_size, UserSettings.DoE_LowerBounds, UserSettings.DoE_UpperBounds);
    DoE.size = UserSettings.DoE_size;
    DoE.vars = UserSettings.DoE_vars;
    DoE.vars(:,8)= {'Imperfection'};
    DoE.LowerBounds = UserSettings.DoE_LowerBounds;
    DoE.UpperBounds = UserSettings.DoE_UpperBounds;
    DoE.points = DoE_points;
    DoE.points(:,8) = UserSettings.RIKS_imperfection;
    % If one of the variables is an integer:
%     DoE.points(:,5) = round(DoE.points(:,5));
    %
    % Save the RVEs_data structure to a MATLAB format:
    savefile = strcat(DoE_dir,'/',UserSettings.DoE_file_name,'.mat');
    save(savefile,'DoE');
else
    disp(' ');
    disp('DoE file already exists! File NOT updated');
    %
%     errorfile_name = strcat(analyses_dir,'/ERROR_FILE'); % Write ERROR file
%     fid = fopen(errorfile_name,'a+');
%     fprintf(fid,'DoE file already exists! File NOT updated\n');
%     fclose(fid);
    loadfile = strcat(DoE_dir,'/',UserSettings.DoE_file_name,'.mat');
    load(loadfile,'DoE');
end
%
% DoE_points(:,8)= DoE.points(:,8);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create or LOAD the Input file (with discrete input properties)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if exist([input_dir,'/',UserSettings.Input_file_name,'.mat'],'file') == 0
    %
    Input.size = UserSettings.Input_size;
    Input.vars = UserSettings.Input_vars;
    Input.points = UserSettings.Input_points;
    %
    % Save the RVEs_data structure to a MATLAB format:
    savefile = strcat(input_dir,'/',UserSettings.Input_file_name,'.mat');
    save(savefile,'Input');
else
    disp(' ');
    disp('INPUT file already exists! File NOT updated');
    %
    loadfile = strcat(input_dir,'/',UserSettings.Input_file_name,'.mat');
    load(loadfile,'Input');
end
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Delete postprocessing files that may have been used previously with
% different CPUs:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if UserSettings.postproc_option == 1
    % Delete postprocessing files for different CPUs (these were just
    % temporary)
    cd(postproc_dir);
%             string = ['rm ',abq_inp_file{jDoE},'*.odb'];
    string = ['rm ',strcat('STRUCTURES_postprocessing_variables_CPU'),'*'];

    system(string);
    %
    cd(main_dir);
end
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Restart MATLAB parallel pool:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% delete(gcp('nocreate'));
% parpool('local', UserSettings.matlab_cpus); % USE WORKERS
%
% Save some of the needed UserSettings to variable (not object oriented),
% just because MATLAB doesn't like it when using parfor:
UserSettings_run_simul_option = UserSettings.run_simul_option;
UserSettings_postproc_option = UserSettings.postproc_option;
DoE_points = DoE.points;
Input_points = Input.points;
%
for iInput=UserSettings.selectInputs
    %
    iInput_dir = strcat(analyses_dir,'/Input_point',int2str(iInput));
    % Create a folder for this set of Fixed Variables, if it doesn't already exist:
    if exist(iInput_dir,'dir') == 0, mkdir(iInput_dir); end
    %
    for kImperfection=UserSettings.selectImperfections
         kImperfection_dir = strcat(iInput_dir,'/Imperfection_point',int2str(kImperfection));
         % Create a folder for this sample of Imperfections, if it doesn't already exist:
         if exist(kImperfection_dir,'dir') == 0, mkdir(kImperfection_dir); end
         %
        % If using parallel for loop uncomment next 3 lines and comment the
        % following 2. Otherwise, do the opposite.
%         parfor jDoE=UserSettings.selectDoEs
%             t = getCurrentTask();
%             CPUid(jDoE) = t.ID;
        for jDoE=UserSettings.selectDoEs
            CPUid(jDoE) = 1;
            %
            % Create a folder for this DoE, if it doesn't already exist:
            jDoE_dir = strcat(kImperfection_dir,'/DoE_point',int2str(jDoE));
            if exist(jDoE_dir,'dir') == 0, mkdir(jDoE_dir); end
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Generates Files with Mesh, PBCs and Material Properties
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %
            abq_inp_file = {};
            file_name = {};
            runfile_name = {};
            %
            %
            abq_inp_file{jDoE} = strcat('DoE',int2str(jDoE)); % file without extension
            runfile_name{jDoE} = strcat(jDoE_dir,'/run_DoE',int2str(jDoE)); % run file for ARES cluster
        %     if UserSettings.analysis_option == 2
        %         abq_inp_file{jDoE} = strcat(abq_inp_file{jDoE},'_implicit_static'); 
        %         runfile_name{jDoE} = strcat(runfile_name{jDoE},'_implicit_static.sh'); % run file for ARES cluster
        %     elseif UserSettings.analysis_option == 3
        %         abq_inp_file{jDoE} = strcat(abq_inp_file{jDoE},'_riks'); % file without extension
        %         runfile_name{jDoE} = strcat(runfile_name{jDoE},'_riks.sh'); % run file for ARES cluster
        %     elseif UserSettings.analysis_option > 3 % Analysis option unavailable!
        % %             disp(strcat('Analysis option selected is unavailable'));
        %         errorfile_name = strcat(jDoE_dir,'/ERROR_FILE');
        %         fid = fopen(errorfile_name,'a+');
        %         fprintf(fid,'\nAnalysis option selected is unavailable');
        %         fclose(fid);
        %         error('Analysis option selected is unavailable');
        %     end
            %
        %     file_name{jDoE} = strcat(jDoE_dir,'/',abq_inp_file{jDoE},'.inp');
            %
            %
            %
            if UserSettings_run_simul_option == 1 % Generate mesh and input files
                % Call function: generate_mesh
                generate_mesh_supercompressible_mm(DoE_points,Input_points,UserSettings,jDoE,kImperfection,iInput,jDoE_dir);
                %
                generate_buckling_simulation_files(DoE_points,Input_points,UserSettings,jDoE,kImperfection,iInput,jDoE_dir,main_dir,abq_inp_file{jDoE});
                %
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Run simulation automatically?                                       %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %     if UserSettings.run_simul_option ~= 0 && UserSettings.analysis_option ~= 1
        %                 if UserSettings.run_simul_option == 1
        %                     cd(jDoE_dir);
        %                     %     system(string);
        %                     % Run simulations in local desktop
        %                     string = ['abaqus job=',abq_inp_file{jDoE},...
        %                         ' cpus=',num2str(ncpus),' double interactive ask_delete=OFF'];
        %                     system(string);
        %                 elseif UserSettings.run_simul_option == 2
        %                     cd(jDoE_dir);
        %                     %                     string = ['chmod +x ',runfile_name{iBC}];
        %                     %                     system(string);
        %                     % Run simulations in ARES cluster
        %                     string = ['qsub ',runfile_name{jDoE}];
        %                     system(string);
        %                 end
        %     end
            %
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Generate Postprocessing file for all RVEs                           %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %
            cd(main_dir); % Return to main folder
            %
            if UserSettings_postproc_option == 1
                % Call function: generate_postproc_file
                generate_buckling_postproc_file(DoE_points,Input_points,UserSettings,jDoE,kImperfection,iInput,jDoE_dir,postproc_dir,CPUid(jDoE));
                %
            end
            %
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Run RIKS simulation and/or do the postprocessing                    %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if UserSettings.RIKS_option == 1
                % Call function:
                generate_riks_results_supercompressible_mm(DoE_points,Input_points,UserSettings,jDoE,kImperfection,iInput,jDoE_dir,postproc_dir,CPUid(jDoE),main_dir,abq_inp_file{jDoE});
                %
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Delete ODB files for this RVE?                                      %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %
            if UserSettings.delodb_option == 1
                % Delete odb for each BC
                cd(jDoE_dir);
        %             string = ['rm ',abq_inp_file{jDoE},'*.odb'];
                string = ['rm ',strcat('DoE',int2str(jDoE)),'*'];

                system(string);
                %
            end
            %
            cd(main_dir); % Return to main folder
            %
            %
        end % end jDoE
        %
    end % end kImperfection
    %
end % end iInput
%
% Since we are using parallel computing, we have to merge the Pickle
% postprocessing files from each CPU:
if UserSettings.postproc_option == 1
    generate_unique_postproc_file(UserSettings,postproc_dir);
end
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Convert the python post-processing file RVEs_postprocessing_variables.p %
% to a MATLAB post-processing file RVEs_postprocessing_variables.m        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
if UserSettings.convert_p2mat_option == 1
    % Read Python homogenized variables from the pickle file generated
    file_postproc = strcat(postproc_dir,'/STRUCTURES_postprocessing_variables.p');
    file_load = loadpickle(file_postproc);
    STRUCTURES_data = file_load.STRUCTURES_data;
    clear file_load;
    % Save the RVEs_data structure to a MATLAB format:
%     savefile = strcat(postproc_dir,'/STRUCTURES_postprocessing_variables.mat');
%     save(savefile,'STRUCTURES_data');
    %
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     % Just check for incomplete/corrupt data in database:
%     icount = 0;
%     icount_selectDoEs = 0;
%     icount_selectInputs = 0;
%     Inputs_strings = fieldnames(STRUCTURES_data);
%     nInputs = numel(Inputs_strings);
%     %
%     for iInput=UserSettings.selectInputs
%         Imperfections_strings = fieldnames(STRUCTURES_data.(['Input',num2str(iInput)]));
%         nImperfections = numel(Imperfections_strings);
%         for kImperfection=UserSettings.selectImperfections
%             incomplete_data_points.(['Input',num2str(iInput)]).(['Imperfection',num2str(kImperfection)]) = UserSettings.selectDoEs;
%         end
%     end
%     %
%     new_selectDoEs = []; % Initialize an empty vector
%     new_selectInputs = []; % Initialize an empty vector
%     unfinished_simulations = {}; % Initialize an empty cell
%     %
%     % Check if each of the output data points was correctly computed:
%     for iInput = 1:nInputs
%         Imperfections_strings = fieldnames(STRUCTURES_data.(Inputs_strings{iInput}));
%         nImperfections = numel(Imperfections_strings);
%         for kImperfection=1:nImperfections
%             DoEs_strings = fieldnames(STRUCTURES_data.(Inputs_strings{iInput}).(Imperfections_strings{kImperfection}));
%             nDoEs = numel(DoEs_strings);
%             for jDoE=1:nDoEs
%                 % Extract just the number of the DoE (not the entire string);
%                 Str = DoEs_strings(jDoE);
%                 Str = Str{1};
%                 Key = 'DoE';
%                 Index = strfind(Str, Key);
%                 jDoE_integer = sscanf(Str(Index(1) + length(Key):end), '%g', 1);
% 
%                 % Eliminate this output data point from the list of output points that are missing
%                 if numel( fieldnames( STRUCTURES_data.(Inputs_strings{iInput}).(Imperfections_strings{kImperfection}).(DoEs_strings{jDoE}) ) ) > 0
%                     incomplete_data_points.(Inputs_strings{iInput}).(Imperfections_strings{kImperfection})( find(incomplete_data_points.(Inputs_strings{iInput}).(Imperfections_strings{kImperfection})==jDoE_integer) ) = [];
%                 end
%             end
%         end
%     end
    %
    % Bad_RVEs =  [new_selectRVEs, RVEs_without_mesh];
    %
    % Save the RVEs_data structure to a MATLAB format:
    savefile = strcat(postproc_dir,'/STRUCTURES_postprocessing_variables.mat');
%     save(savefile,'STRUCTURES_data','incomplete_data_points','UserSettings');
    %
    %
end
%
disp('Total Elapsed Time [min]: ');
disp(toc/60);
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of MAIN Routine                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
