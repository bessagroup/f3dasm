import numpy as np
import os
import pickle

from f3dasm.simulator.abaqus.run.run_sim import execute_abaqus


def run_job_from_inp(inp_file, sim_dir):
    initial_wd  = os.getcwd()
    os.chdir(sim_dir)
    command = 'abaqus job={}'.format(inp_file)
    os.system(command)
    os.chdir(initial_wd)


from f3dasm.doe.doevars import  DoeVars
from f3dasm.doe.sampling import SalibSobol, NumpyLinear

from f3dasm.doe.sampling import SamplingMethod
from numpy.core.records import array
import numpy as np
from f3dasm.simulator.abaqus.run.run_sim import _create_temp_dir


vars = {'ratio_d': 0.006, #[0.004, 0.073],
        'ratio_pitch': [0.75, 0.9],  #[.25, 1.5],
        'ratio_top_diameter': 0.7, #[0., 0.8],
            'n_longerons': 3,      
            'bottom_diameter': 100.,
            'young_modulus': 3500.,
            'shear_modulus': 1287., 
            'imperfections' : 0.001 }

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




class AbaqusProcess():


    def __init__(self, name, 
                config_filename, 
                abq_script = None, 
                ):



        self.config = {}
        self.config['name'] = name
        self.config['abaqus_script'] = abq_script

        self.config_filename = config_filename
    

    def write_input_pkl(self,simdir, inputs = None):
        filename =  os.path.join(simdir, self.config_filename)
        data = {}
        data['config'] = self.config
        data['variables'] = inputs

        with open(filename, 'wb') as file:
            pickle.dump(data, file, protocol = 2)

    def execute_abq(self):
        run_module_name = 'f3dasm.simulator.abaqus.abaqus_src.run.run_from_inp'
        abaqus_path='abaqus'
        temp_dir_name = '_temp'
        execute_abaqus(run_module_name,  
                        i_doe_path, temp_dir_name, #temp_dir_name, 
                        abaqus_path =abaqus_path, gui = False)

class PostProc():

    def __init__(self, name, 
                    abq_script,
                    keep_odb = True,  
                    config = {}):
        self.name = name
        self.config = config
        self.config['name'] = name
        self.config['post_processing_fnc'] = abq_script
        self.config['keep_odb'] = keep_odb

    def write_input_pkl(self, filename = 'postproc_config.pkl'):
        data = {}
        data['config'] = self.config

        with open(filename, 'wb') as file:
            pickle.dump(data, file, protocol = 2)

    def execute(self):
        self.write_input_pkl()
        return


class Preprocessing():

    def __init__(self, name, 
                    abq_script, 
                    config = {}):
        self.name = name
        self.config = config
        self.config['name'] = name
        self.config['abaqus_script'] = abq_script

    def write_input_pkl(self, inputs, filename = 'preproc_inputs.pkl'):
        data = {}
        data['config'] = self.config
        data['variables'] = inputs

        with open(filename, 'wb') as file:
            pickle.dump(data, file, protocol = 2)

    def execute(self, inputs):
        self.write_input_pkl(inputs)

        run_module_name = 'f3dasm.simulator.abaqus.abaqus_src.run.run_from_inp'
        abaqus_path='abaqus'
        temp_dir_name = '_temp'
        execute_abaqus(run_module_name,  
                        i_doe_path, temp_dir_name, #temp_dir_name, 
                        abaqus_path =abaqus_path, gui = False)
        return


class RunJob():
    def __init__(self, name, 
                    config = {}):
        self.name = name
        self.config = config
        self.config['name'] = name

    def write_input_pkl(self, filename = 'sim_config.pkl'):
        data = {}
        data['config'] = self.config

        with open(filename, 'wb') as file:
            pickle.dump(data, file, protocol = 2)

    def execute(self, inputs):
        self.write_input_pkl(inputs)
        return



preproc_1 = Preprocessing( name = 'lin_bckl', 
                    abq_script = 'abaqus_modules.supercompressible_fnc.lin_buckle', 
                    config = {})

sim1 = RunJob(name ='lin_bckl' )
postproc1 = PostProc(name ='lin_bckl',
                    abq_script  = 'abaqus_modules.supercompressible_fnc.post_process_lin_buckle')

sim_list = [preproc_1]

#sim_list  = ['lin_bckl', 'riks']


example_name = 'example_3'

if not os.path.exists(example_name):
    #raise Exception('Name already exists')
    os.mkdir(example_name)
analysis_folder  = os.path.join(example_name, 'analyses')
os.mkdir(analysis_folder )



temp_dir_name = '_temp'
_create_temp_dir(temp_dir_name)

for sim in sim_list:
    sim_dir = os.path.join(analysis_folder, sim.name )
    os.mkdir(sim_dir)

    for i_doe in doe_list:
        i_doe_path = os.path.join(sim_dir,  'DoE_point%i' % i_doe)
        os.mkdir( i_doe_path)

        inputs = doe_pd.iloc[i_doe].to_dict()
        inputs['n_longerons'] = int(inputs['n_longerons'])

        preproc_filename = os.path.join(i_doe_path, 'preproc_inputs.pkl')
        preproc_1.write_input_pkl(inputs, preproc_filename)

        #preproc:
        run_module_name = 'f3dasm.simulator.abaqus.abaqus_src.pre_process.preproc'
        abaqus_path='abaqus'
        

        sim_filename = os.path.join(i_doe_path, 'sim_config.pkl')
        sim1.write_input_pkl(sim_filename)

        ppc_filename = os.path.join(i_doe_path, 'postproc_config.pkl')
        postproc1.write_input_pkl(ppc_filename)

        #temp_dir_name = os.path.join(i_doe_path, temp_dir_name )
        #if not os.path.exists(temp_dir_name):
        #    os.mkdir(temp_dir_name)

        #temp_dir = os.path.join(i_doe_path, temp_dir_name )
        #if not os.path.exists(temp_dir):
            #raise Exception('Name already exists')
        #    os.mkdir(temp_dir)
        #temp_dir
        execute_abaqus(run_module_name, i_doe_path, temp_dir_name, #temp_dir_name, 
                        abaqus_path =abaqus_path, gui = False)

    
        # scratch_dir = os.path.join(i_doe_path, temp_dir_name )
        # if not os.path.exists(scratch_dir):
        #     os.mkdir(scratch_dir)


        #run:
        run_module_name = 'f3dasm.simulator.abaqus.abaqus_src.run.run_from_inp'
        abaqus_path='abaqus'
        temp_dir_name = '_temp'
        execute_abaqus(run_module_name,  
                        i_doe_path, temp_dir_name, #temp_dir_name, 
                        abaqus_path =abaqus_path, gui = False)


        #postproc:
        run_module_name = 'f3dasm.simulator.abaqus.abaqus_src.post_processing.post_proc'
        abaqus_path='abaqus'
        temp_dir_name = '_temp'
        execute_abaqus(run_module_name,  
                        i_doe_path, temp_dir_name, #temp_dir_name, 
                        abaqus_path =abaqus_path, gui = False)
        #if execute_directly:
            #inp_file_name = os.path.join(sim.name + '_job')
            #run_job_from_inp(inp_file_name, sim_dir =i_doe_path )






# sim_dir = os.path.join(analysis_folder, sim_list[0].name)
# sim_dir = os.path.join(sim_dir, 'DoE_point%i' % i_doe)


# #preproc:
# run_module_name = 'f3dasm.simulator.abaqus.abaqus_src.run.preproc'
# abaqus_path='abaqus'
# temp_dir_name = '_temp'
# _create_temp_dir(temp_dir_name)


# execute_abaqus(run_module_name, sim_dir, temp_dir_name, 
#                 abaqus_path =abaqus_path, gui = False)

print(123)
# def _read_data():

#     # get pickle filename
#     filename = get_unique_file_by_ext(ext='.pkl')

#     # read file
#     with open(filename, 'rb') as file:
#         data = convert_dict_unicode_str(pickle.load(file))

#     return filename, data


# class Step():

#     def __init__(self) -> None:
#         pass 

#     def execute(self):
#         pass
        


# class PreProcessAbaqus():

#     def __init__(self, 
#                 preprocessing_script = None) -> None:
#         self.preprocessing_script = preprocessing_script


#         abstract_model = import_abstract_obj(info['abstract_model'])
#         pp_fnc_loc = info.get('post_processing_fnc', None)
#         post_processing_fnc = import_abstract_obj(pp_fnc_loc) if pp_fnc_loc is not None else None
#         for key in ['abstract_model', 'post_processing_fnc']:
#             info.pop(key, None)

#         # get args
#         args = self.variables.copy()
#         args.update(info)

#         # instantiate model
#         if i:
#             args.update({'previous_model': list(self.models.values())[i - 1]})
#         if issubclass(abstract_model, BasicModel):
#             model = abstract_model(name=model_name, **args)
#         else:
#             model = WrapperModel(name=model_name, abstract_model=abstract_model,
#                                     post_processing_fnc=post_processing_fnc,
#                                      **args)

#             self.models[model_name] = model

#     def execute()



# class Simulation():

#     def __init__(self, 
#                 pre_process, 
#                 run_sim, 
#                 post_process):


#     def execute(self, inputs):

#         return results
