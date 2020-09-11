'''
Created on 2020-04-22 14:53:01
Last modified on 2020-09-11 17:03:32
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
import glob
import pickle
import importlib
from collections import OrderedDict
import time
import traceback

# local library
from .run_utils import convert_dict_unicode_str
from ..modelling.model import BasicModel
from ..modelling.model import WrapperModel
from f3das.utils.file_handling import get_unique_file_by_ext


#%% object definition

# TODO: add error file

# TODO: run model that takes into account F3DAS and non-F3DAS scripts
# TODO: need to understand where to put non-F3DAS scripts due to F3DAS installation


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
        self.run_time = None
        self.post_processing_time = None
        # read data
        data = self._read_data()
        self.pickle_dict = data
        # store variables
        # TODO: transform inputs here?
        self.variables = data['variables']
        self.sim_info = data['sim_info']
        self.keep_odb = data['keep_odb']
        self.dump_py_objs = data['dump_py_objs']
        n_sims = len(self.sim_info)
        self.abstract_models = self._import_abstract_models(
            data['abstract_model'], n_sims)
        self.post_processing_fncs = self._import_post_processing_fncs(
            data.get('post_processing_fnc', None), n_sims)
        # initialize variables
        self.models = OrderedDict()
        self.post_processing = OrderedDict()

    def execute(self):

        # TODO: better control of running time

        # instantiate models
        self._instantiate_models()

        # run models
        run_time_init = time.time()
        self._run_models()
        self.run_time = time.time() - run_time_init

        # post-processing
        pp_time_init = time.time()
        self._perform_post_processing()
        self.post_processing_time = time.time() - pp_time_init

        # dump results
        self._dump_results()

        # delete unnecessary files
        self._clean_dir()

    def _read_data(self):

        # get pickle filename
        self.filename = get_unique_file_by_ext(ext='.pkl')

        # read file
        with open(self.filename, 'rb') as file:
            data = convert_dict_unicode_str(pickle.load(file))

        return data

    def _import_abstract_models(self, abstract_models, n_sims):

        if type(abstract_models) is str:
            abstract_models = [abstract_models] * n_sims

        abstract_models_ = []
        for abstract_model in abstract_models:
            module_name, method_name = abstract_model.rsplit('.', 1)
            module = importlib.import_module(module_name)
            abstract_models_.append(getattr(module, method_name))

        return abstract_models_

    def _import_post_processing_fncs(self, post_processing_fncs, n_sims):

        if post_processing_fncs is None:
            return [None] * n_sims

        if type(post_processing_fncs) is str:
            post_processing_fncs = [post_processing_fncs] * n_sims

        post_processing_fncs_ = []
        for post_processing_fnc in post_processing_fncs:
            module_name, method_name = post_processing_fnc.rsplit('.', 1)
            module = importlib.import_module(module_name)
            post_processing_fncs_.append(getattr(module, method_name))

        return post_processing_fncs_

    def _instantiate_models(self):

        for i, (abstract_model, (model_name, info)) in enumerate(zip(self.abstract_models, self.sim_info.items())):

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
                                     post_processing_fnc=self.post_processing_fncs[i],
                                     **args)

            self.models[model_name] = model

    def _run_models(self):

        for model in self.models.values():
            model.create_model()
            model.write_inp(submit=True)

    def _perform_post_processing(self):

        for model_name, model in reversed(self.models.items()):

            # avoid doing again post-processing
            if model_name in self.post_processing.keys():
                continue

            # do post-processing of current model
            if isinstance(model, WrapperModel) and not callable(model.post_processing_fnc):
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
        self.pickle_dict['time'] = {'total_time': time.time() - self.init_time,
                                    'run_time': self.run_time,
                                    'post_processing_time': self.post_processing_time}
        # TODO: update success to be less permissive (e.g. subroutine location)
        self.pickle_dict['success'] = True
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
        filename = get_unique_file_by_ext(ext='.pkl')
        run_model.pickle_dict['success'] = False
        with open(filename, 'wb') as file:
            data = pickle.dump(run_model.pickle_dict, file, protocol=2)
