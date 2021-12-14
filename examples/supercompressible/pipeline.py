import os
import pickle
import time
from f3dasm.simulator.abaqus.run.run_sim import execute_abaqus


def run_job_from_inp(inp_file, sim_dir):
    initial_wd  = os.getcwd()
    os.chdir(sim_dir)
    command = 'abaqus job={}'.format(inp_file)
    os.system(command)
    os.chdir(initial_wd)

import shutil
from f3dasm.doe.doevars import  DoeVars
import numpy as np
from f3dasm.simulator.abaqus.run.run_sim import _create_temp_dir


vars = {'ratio_d': 0.006, #[0.004, 0.073],
        'ratio_pitch': [0.75, 0.9],  #[.25, 1.5],
        'ratio_top_diameter': 0.7, #[0., 0.8],
            'n_longerons': 3,      
            'bottom_diameter': 100.,
            'young_modulus': 3500.,
            'shear_modulus': 1287.}

doe = DoeVars(vars)
print('DoEVars definition:')
print(doe)

print('\n DoEVars summary information:')
print(doe.info())

# Compute sampling and combinations
doe.do_sampling()

print('\n Pandas dataframe with compbined-sampled values:')
print(doe.data)

doe_pd = doe.data
doe_list = doe_pd.index.values.tolist()

class AbaqusStep():
    def __init__(self, name, 
                config_filename, 
                abq_script = None, 
                abq_run_module = None, 
                ):

        self.config_filename = config_filename 
        self.run_module_name = abq_run_module  #'f3dasm.simulator.abaqus.abaqus_src.run.run_from_inp'
        self.abaqus_path='abaqus'
        self.temp_dir_name = '_temp'
        self.config = {}
        self.config['name'] = name
        #job info field passes kwargs to Abaqus mdb.JobFromInputFile(), for all possible kwargs check:
        #https://abaqus-docs.mit.edu/2017/English/SIMACAEKERRefMap/simaker-c-jobfrominputfilepyc.htm
        self.config['job_info'] = {'name': name,   #abaqus resires at least name, other kwargs are optional
                                    'numCpus' : 1, 
                                    #'userSubroutine': "" ,  # A WAY TO ADD USER SUBROUTINE
                                    }

        if abq_script is not None:
            self.config['abq_script'] = abq_script

    def write_input_pkl(self, simdir, inputs = None):
        filename =  os.path.join(simdir, self.config_filename)
        data = {}
        data['config'] = self.config
        data['variables'] = inputs

        with open(filename, 'wb') as file:
            pickle.dump(data, file, protocol = 2)

    def execute(self, simdir, inputs = None):
        inpt_file = os.path.join(simdir, self.config_filename)
        if not os.path.exists(inpt_file):
            self.write_input_pkl(simdir, inputs = inputs)

        execute_abaqus(self.run_module_name,  
                        simdir, self.temp_dir_name,
                        abaqus_path =self.abaqus_path, gui = False)

class PostProc(AbaqusStep):
    def __init__(self, name , abq_script, 
                keep_odb = True):
        config_filename = 'postproc_config.pkl'
        abq_run_module=  'f3dasm.simulator.abaqus.abaqus_src.post_processing.post_proc'
        
        super().__init__(name, config_filename, 
                        abq_script =abq_script, 
                        abq_run_module=abq_run_module)

        self.config['keep_odb'] = keep_odb

class PreProc(AbaqusStep):
    def __init__(self, name, abq_script):
        config_filename = 'preproc_inputs.pkl'
        abq_run_module = 'f3dasm.simulator.abaqus.abaqus_src.pre_process.preproc'

        super().__init__(name, config_filename, abq_script=abq_script, abq_run_module=abq_run_module)


class RunJob(AbaqusStep):

    def __init__(self, name):
        config_filename ='sim_config.pkl'
        abq_run_module= 'f3dasm.simulator.abaqus.abaqus_src.run.run_from_inp'
        super().__init__(name, config_filename,
                        abq_run_module=abq_run_module)


class Simulation():
    def __init__(self, name, 
                preproc_script = None, 
                postproc_script = None,
               ): 
        self.name = name
        self.preproc = PreProc(name = name, abq_script =preproc_script)
        self.job = RunJob(name)
        self.postproc = PostProc(name = name, abq_script = postproc_script)

    def write_configs(self, simdir, inputs = None):
        self.preproc.write_input_pkl(simdir = simdir, inputs = inputs )
        self.job.write_input_pkl(simdir = simdir)
        self.postproc.write_input_pkl(simdir = simdir)

    def execute(self, simdir, inputs):
        self.preproc.execute(simdir = simdir, inputs=inputs)
        self.job.execute(simdir = simdir)
        self.postproc.execute(simdir = simdir)

    def extract_results(self, simdir):
        file_name = self.name + '_postproc'
        file_name = os.path.join(simdir, file_name)
        with open(file_name, 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        return data

def get_inputs_riks(inputs, sim_lin_bckl, i_doe_lin_buckle_path):
    data_lin_buckle = sim_lin_bckl.extract_results(simdir=i_doe_lin_buckle_path )
    inputs_riks = inputs.copy()
    inputs_riks['coilable'] = int(data_lin_buckle['post-processing']['coilable'])
    inputs_riks['lin_bckl_max_disp'] = data_lin_buckle['post-processing']['max_disps'][1]
    inputs_riks['lin_buckle_odb'] = sim_lin_bckl.name
    inputs_riks['imperfection'] = 0.001
    return inputs_riks

example_name = 'example_4'

if not os.path.exists(example_name):
    os.mkdir(example_name)
analysis_folder  = os.path.join(example_name, 'analyses')
os.mkdir(analysis_folder )



sim_lb = Simulation(name = 'linear_buckle', 
                preproc_script =  'abaqus_modules.supercompressible_fnc.lin_buckle', 
                postproc_script = 'abaqus_modules.supercompressible_fnc.post_process_lin_buckle'
                )

sim_riks = Simulation(name = 'riks', 
                preproc_script =  'abaqus_modules.supercompressible_fnc.riks', 
                postproc_script = 'abaqus_modules.supercompressible_fnc.post_process_riks'
                )
sim_riks.job.config['job_info']['numCpus'] = 1
sim_lb.job.config['job_info']['numCpus'] = 1


temp_dir_name = '_temp'
_create_temp_dir(temp_dir_name)

sim_lb_path = os.path.join(analysis_folder, sim_lb.name )
os.mkdir(sim_lb_path)

sim_rx_path = os.path.join(analysis_folder, sim_riks.name )
os.mkdir(sim_rx_path)


for i_doe in doe_list:

    #LINEAR BUCKLING
    i_doe_path = os.path.join(sim_lb_path,  'DoE_point%i' % i_doe)
    os.mkdir( i_doe_path)
    inputs = doe_pd.iloc[i_doe].to_dict()
    inputs['n_longerons'] = int(inputs['n_longerons'])
    sim_lb.execute(simdir=i_doe_path, inputs = inputs)

    #RIKS    
    inputs_riks = get_inputs_riks(inputs, sim_lb, i_doe_path)    
    i_doe_riks = os.path.join(sim_rx_path,  'DoE_point%i' % i_doe)
    os.mkdir( i_doe_riks)

    #Riks needs access to lin buckle odb file, 
    lb_odb = os.path.join(i_doe_path, sim_lb.name + '.odb')
    target = os.path.join(i_doe_riks, sim_lb.name + '.odb')
    shutil.copyfile(lb_odb, target, follow_symlinks=True)
    while not os.path.exists(target):
        print('odb, sleepin')
        time.sleep(0.001)
    #with odb files we also need to pass prt file, in order for odb to recognize the model instance
    lb_odb = os.path.join(i_doe_path, sim_lb.name + '.prt')
    target = os.path.join(i_doe_riks, sim_lb.name + '.prt')
    shutil.copyfile(lb_odb, target, follow_symlinks=True)
    while not os.path.exists(target):
        print('inp, sleepin')
        time.sleep(0.001)


    sim_riks.write_configs(simdir = i_doe_riks, inputs = inputs_riks)
    sim_riks.execute(simdir = i_doe_riks, inputs = inputs_riks)

    riks_data = sim_riks.extract_results(i_doe_riks)
