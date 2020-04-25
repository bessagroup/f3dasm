'''
Created on 2020-04-22 14:53:01
Last modified on 2020-04-25 16:33:23
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


#%% imports

# abaqus
from abaqus import session

# standard library
import os
import pickle
import importlib
from collections import OrderedDict
import time

# local library
from .misc import convert_dict_unicode_str


#%% object definition

class RunModel(object):

    def __init__(self, filename=''):
        '''

        Parameters
        ----------
        filename : str
            Pickle filename. If empty, then assumes there's only one pickle
            file in the directory.

        Notes
        -----
        -assumes the same data is required to instantiate each model of the
        sequence.
        '''
        # read data
        self.filename = filename
        data = self._read_data(filename)
        self.pickle_dict = data
        # store variables
        self.abstract_model = self._import_abstract_model(data['abstract_model'])
        self.data = data['data']
        self.sim_info = data['sim_info']
        self.init_time = time.time()
        # initialize variables
        self.models = OrderedDict()
        self.post_processing = OrderedDict()

    def execute(self):

        # run models
        self._run_models()

        # post-processing
        self._perform_post_processing()

        # dump results
        self._dump_results()

    def _read_data(self, filename):

        # get pickle filename
        if not filename:
            for fname in os.listdir(os.getcwd()):
                if fname.endswith('.pkl'):
                    self.filename = fname
                    break

        # read file
        with open(self.filename, 'rb') as file:
            data = convert_dict_unicode_str(pickle.load(file))

        return data

    def _import_abstract_model(self, abstract_model):

        module_name, method_name = abstract_model.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, method_name)

    def _run_models(self):

        for i, (model_name, info) in enumerate(self.sim_info.items()):

            # get args
            args = self.data.copy()
            args.update(info)

            # instantiate model
            previous_model = list(self.models.values())[i - 1] if i else None
            model = self.abstract_model(name=model_name,
                                        previous_model=previous_model, **args)

            # create and run model
            model.create_model()
            model.write_inp(submit=True)

            # dump and save model
            model.dump(create_file=False)
            self.models[model_name] = model

    def _perform_post_processing(self):

        for model_name, model in reversed(self.models.items()):

            # avoid doing again post-processing
            if model_name in self.post_processing.keys():
                continue

            # do post-processing of current model
            odb_name = '%s.odb' % model.job_name
            odb = session.openOdb(name=odb_name)
            self.post_processing[model_name] = model.perform_post_processing(odb)

            # save post-processing of previous model (if applicable)
            if model.previous_model_results is not None and model.previous_model.name not in self.post_processing.keys():
                self.post_processing[model.previous_model.name] = model.previous_model_results
        self.post_processing = OrderedDict(reversed(list(self.post_processing.items())))

    def _dump_results(self):

        # results readable outside abaqus
        self.pickle_dict['post-processing'] = self.post_processing
        self.pickle_dict['time'] = time.time() - self.init_time
        with open(self.filename, 'wb') as file:
            pickle.dump(self.pickle_dict, file, protocol=2)

        # more complete results readable within abaqus
        self.pickle_dict['models'] = self.models
        with open('%s_abaqus' % self.filename, 'wb') as file:
            pickle.dump(self.pickle_dict, file, protocol=2)


if __name__ == '__main__':

    # create run model
    run_model = RunModel()

    # run models
    run_model.execute()
