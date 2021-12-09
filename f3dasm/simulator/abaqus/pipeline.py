import numpy as np

from .utils import convert_dict_unicode_str
from .stats import get_wait_time_from_log
from ..modelling.model import BasicModel
from ..modelling.model import WrapperModel
from .....utils.file_handling import get_unique_file_by_ext
from .....utils.utils import import_abstract_obj

import pickle



class ABQStep():

    def __init__(self, 
                abq_script, 
                name):
        self.config = {}

    # def write_step_config(self):
    #     self.config

    def write_input_pkl(self, inputs):
        data = {}
        data['config'] = self.config
        data['inputs'] = inputs


    def execute(self, inputs):
        self.write_input_pkl(inputs)
        return

class ABQPreproc(ABQStep):
    def __init__(self, abq_script, name):
        super().__init__(abq_script, name)












def _read_data():

    # get pickle filename
    filename = get_unique_file_by_ext(ext='.pkl')

    # read file
    with open(filename, 'rb') as file:
        data = convert_dict_unicode_str(pickle.load(file))

    return filename, data


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
