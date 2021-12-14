import os 
import pickle
from f3dasm.simulator.abaqus.abaqus_src import abaqus_module_call

def execute_abaqus(run_module_name, sim_dir, temp_dir_name, 
                    abaqus_path = 'abaqus', 
                    gui = False, 
                    execute_script = os.path.abspath(abaqus_module_call.__file__) #location of the caller script
                    ):
    """
    This function constructs a command for 'abaqus cae' in order to execute an abaqus module. 
    The command passes abaqus module name (run_module_name) and directories to a generic 
    abaqus scipt (execute_script). This script is then executed with the internal abaqus python 
    interpreter. 
    """
    gui_ = 'script' if gui else 'noGUI'
    command = '{} cae {}={} -- -func {} -sdir {} -tdir {}'.format(abaqus_path, gui_, 
                                                                execute_script, 
                                                                run_module_name, 
                                                                sim_dir, 
                                                                temp_dir_name)
    os.system(command)


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
        abq_run_module=  'f3dasm.simulator.abaqus.abaqus_src.run_modules.post_proc'
        
        super().__init__(name, config_filename, 
                        abq_script =abq_script, 
                        abq_run_module=abq_run_module)

        self.config['keep_odb'] = keep_odb

class PreProc(AbaqusStep):
    def __init__(self, name, abq_script):
        config_filename = 'preproc_inputs.pkl'
        abq_run_module = 'f3dasm.simulator.abaqus.abaqus_src.run_modules.preproc'

        super().__init__(name, config_filename, abq_script=abq_script, abq_run_module=abq_run_module)


class RunJob(AbaqusStep):

    def __init__(self, name):
        config_filename ='sim_config.pkl'
        abq_run_module= 'f3dasm.simulator.abaqus.abaqus_src.run_modules.run_from_inp'
        super().__init__(name, config_filename,
                        abq_run_module=abq_run_module)
