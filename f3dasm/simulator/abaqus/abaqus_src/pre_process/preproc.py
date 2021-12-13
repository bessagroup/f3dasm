import pickle
import traceback

# local library
from ..utils import convert_dict_unicode_str
from ..modelling.model import BasicModel
from ..modelling.model import WrapperModel
from .....utils.utils import import_abstract_obj



class PreProc(object):

    def __init__(self):
        '''
        wrapper for executing abq py script for preprocesing
        '''
        self.filename, data = _read_data()
        self.variables = data['variables']
        self.sim_info = data['config']
        model_name = self.sim_info['name']

        job_name =  model_name
        job_info  = {'job_info' : {'name' : job_name}}
        abstract_model = import_abstract_obj(self.sim_info['abaqus_script'])

        args = self.variables.copy()
        args.update(job_info)

        if issubclass(abstract_model, BasicModel):
            model = abstract_model(name=model_name,   **args)
        else:
            model = WrapperModel(name=model_name, abstract_model=abstract_model,
                                   **args)
        self.model = model

    def execute(self):
        self.model.create_model()
        self.model.write_inp(submit=False)


def _read_data():
    filename = 'preproc_inputs.pkl'
    with open(filename, 'rb') as file:
        data = convert_dict_unicode_str(pickle.load(file))
    return filename, data

if __name__ == '__main__':

    try: 
        preprocessor = PreProc()
        preprocessor.execute()
    except:

        with open('error.txt', 'w') as file:
            traceback.print_exc(file=file)
        # update success flag
        filename, data = _read_data()
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        data['success'] = False
        with open(filename, 'wb') as file:
            data = pickle.dump(data, file, protocol=2)
