'''
Created on 2020-04-22 14:53:01
Last modified on 2020-09-30 14:37:32
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Run a model sequentially.


Notes
-----
-abaqus cae is kept open during simulation time (appropriate if running time
is low).
'''


# imports

# abaqus
#from abaqus import session

# standard library
#import os
#import glob
import pickle
from collections import OrderedDict
#import time
import traceback

# local library
from ..utils import convert_dict_unicode_str
#from .stats import get_wait_time_from_log
from ..modelling.model import BasicModel
from ..modelling.model import WrapperModel
from .....utils.file_handling import get_unique_file_by_ext
from .....utils.utils import import_abstract_obj



class PreProc(object):

    def __init__(self):
        '''
        Notes
        -----
        -assumes the same data is required to instantiate each model of the
        sequence.
        '''

        # read data
        self.filename, data = _read_data()
        #self.pickle_dict = data

        # store variables
        self.variables = data['variables']
        self.sim_info = data['sim_info']
        model_name, info = self.sim_info.items()

            # abstract objects
        abstract_model = import_abstract_obj(info['abstract_model'])

        args = self.variables.copy()
        args.update(info)

        if issubclass(abstract_model, BasicModel):
            model = abstract_model(name=model_name, **args)
        else:
            model = WrapperModel(name=model_name, abstract_model=abstract_model,
                                    #post_processing_fnc=post_processing_fnc,
                                    **args)
        self.model = model

    def execute(self):
        self.model.create_model()
        self.model.write_inp(submit=False)




def _read_data():

    # get pickle filename
    filename = get_unique_file_by_ext(ext='.pkl')

    # read file
    with open(filename, 'rb') as file:
        data = convert_dict_unicode_str(pickle.load(file))

    return filename, data


if __name__ == '__main__':

    try:  # to avoid to stop running due to one simulation error
        # create run model
        preproseccor = PreProc()
        # run models
        preproseccor.execute()

    except:

        # create error file
        with open('error.txt', 'w') as file:
            traceback.print_exc(file=file)

        # update success flag
        filename, data = _read_data()
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        data['success'] = False
        with open(filename, 'wb') as file:
            data = pickle.dump(data, file, protocol=2)
