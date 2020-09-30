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
from abaqus import session

# standard library
import os
import glob
import pickle
from collections import OrderedDict
import time
import traceback

# local library
from .utils import convert_dict_unicode_str
from .stats import get_wait_time_from_log
from ..modelling.model import BasicModel
from ..modelling.model import WrapperModel
from ...utils.file_handling import get_unique_file_by_ext
from ...utils.utils import import_abstract_obj


# object definition

class RunModel(object):

    def __init__(self):
        '''
        Notes
        -----
        -assumes the same data is required to instantiate each model of the
        sequence.
        '''
        # performance related
        self.init_time = time.time()
        self.time = OrderedDict([('total', None),
                                 ('running', OrderedDict()),
                                 ('waiting', OrderedDict()),
                                 ('post_processing', None)])
        # read data
        self.filename, data = _read_data()
        self.pickle_dict = data
        # store variables
        self.variables = data['variables']
        self.sim_info = data['sim_info']
        self.keep_odb = data['keep_odb']
        self.dump_py_objs = data['dump_py_objs']
        # initialize variables
        self.models = OrderedDict()
        self.post_processing = OrderedDict()

    def execute(self):

        # instantiate models
        self._instantiate_models()

        # run models
        self._run_models()

        # post-processing
        start_time = time.time()
        self._perform_post_processing()
        self.time['post_processing'] = time.time() - start_time

        # dump results
        self._dump_results()

        # delete unnecessary files
        self._clean_dir()

    def _instantiate_models(self):

        for i, (model_name, info) in enumerate(self.sim_info.items()):

            # abstract objects
            abstract_model = import_abstract_obj(info['abstract_model'])
            pp_fnc_loc = info.get('post_processing_fnc', None)
            post_processing_fnc = import_abstract_obj(pp_fnc_loc) if pp_fnc_loc is not None else None
            for key in ['abstract_model', 'post_processing_fnc']:
                info.pop(key, None)

            # get args
            args = self.variables.copy()
            args.update(info)

            # instantiate model
            if i:
                args.update({'previous_model': list(self.models.values())[i - 1]})
            if issubclass(abstract_model, BasicModel):
                model = abstract_model(name=model_name, **args)
            else:
                model = WrapperModel(name=model_name, abstract_model=abstract_model,
                                     post_processing_fnc=post_processing_fnc,
                                     **args)

            self.models[model_name] = model

    def _run_models(self):

        for model in self.models.values():
            start_time = time.time()

            # run and create model
            model.create_model()
            model.write_inp(submit=True)

            # store times
            wait_time = get_wait_time_from_log(model.job_info['name']) if not model.abort else 0.
            self.time['running'][model.name] = time.time() - start_time - wait_time
            self.time['waiting'][model.name] = wait_time

    def _perform_post_processing(self):

        for model_name, model in reversed(self.models.items()):

            # avoid doing again post-processing
            if model_name in self.post_processing.keys():
                continue

            # do post-processing of current model
            if isinstance(model, WrapperModel) and not callable(model.post_processing_fnc) or model.abort:
                self.post_processing[model_name] = None
            else:

                odb_name = '%s.odb' % model.job_info['name']
                odb = session.openOdb(name=odb_name)
                self.post_processing[model_name] = model.perform_post_processing(odb)
                odb.close()

            # save post-processing of previous model (if applicable)
            if model.previous_model_results is not None and model.previous_model.name not in self.post_processing.keys():
                self.post_processing[model.previous_model.name] = model.previous_model_results
        self.post_processing = OrderedDict(reversed(list(self.post_processing.items())))

    def _dump_results(self):

        # results readable outside abaqus
        self.pickle_dict['post-processing'] = self.post_processing
        # TODO: update success to be less permissive (e.g. subroutine location)
        self.pickle_dict['success'] = True
        self.time['total'] = time.time() - self.init_time
        self.pickle_dict['time'] = self.time
        with open(self.filename, 'wb') as file:
            pickle.dump(self.pickle_dict, file, protocol=2)

        # more complete results readable within abaqus
        if self.dump_py_objs:
            # prepare models to be dumped
            for model in self.models.values():
                model.dump(create_file=False)
            # dump models
            with open('%s_abq' % self.filename, 'wb') as file:
                pickle.dump(self.models, file, protocol=2)

    def _clean_dir(self):
        # return if is to keep odb
        if self.keep_odb:
            return

        job_names = [model.job_name for model in self.models.values()]
        for name in job_names:
            for filename in glob.glob('%s*' % name):
                if not filename.endswith(('.pkl', '.pkl_abq')):
                    try:
                        os.remove(filename)
                    except:
                        pass


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
        run_model = RunModel()

        # run models
        run_model.execute()

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
