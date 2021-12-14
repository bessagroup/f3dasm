from abaqus import session
import os
import glob
import pickle
import traceback

from ..utils import convert_dict_unicode_str
from .....utils.utils import import_abstract_obj


class PostProc(object):

    '''
    postproc_config = {
        'simulation_name' : 'string -> module.location.abqus_src.'
        'post_processing_fnc' : 'string', 
        'keep_odb' : True/False
    }
    '''

    def __init__(self):
        data = self.load_postproc_config() #the config variables are loaded from pkl file
        self.config = data['config']
        self.output = {}


    def load_postproc_config(self):
        filename = 'postproc_config.pkl'
        with open(filename, 'rb') as file:
            data = convert_dict_unicode_str(pickle.load(file))
        return data

    def postproc_odb(self) :
        odb_name = '%s.odb' % self.config['name']
        post_processing_fnc = import_abstract_obj(self.config['abq_script'])
        odb = session.openOdb(name=odb_name)
        post_proc_data = post_processing_fnc(odb)
        odb.close()

        self.output['post-processing'] = post_proc_data


    def _dump_results(self):

        # results readable outside abaqus
        # TODO: update success to be less permissive (e.g. subroutine location)
        self.output['success'] = True
        filename =  self.config['name'] + '_postproc'
        with open(filename, 'wb') as file:
            pickle.dump(self.output, file, protocol=2)

        # more complete results readable within abaqus
        # if self.dump_py_objs:
        #     # prepare models to be dumped
        #     for model in self.models.values():
        #         model.dump(create_file=False)
        #     # dump models
        #     with open('%s_abq' % self.filename, 'wb') as file:
        #         pickle.dump(self.models, file, protocol=2)

    def _clean_dir(self):
        # return if is to keep odb
        if self.config['keep_odb']:
            return

        job_name = self.config['name']
        for filename in glob.glob('%s*' % job_name):
            if not filename.endswith(('.pkl', '.pkl_abq')):
                try:
                    os.remove(filename)
                except:
                    pass


    def execute(self):
        self.postproc_odb()
        self._dump_results()
        self._clean_dir()


if __name__ == '__main__':
    
    try:  # to avoid to stop running due to one simulation error

        results = PostProc()
        results.execute()

    except:

        # create error file
        with open('postproc_error.txt', 'w') as file:
            traceback.print_exc(file=file)

        # update success flag
        results.output['success'] = False
        results._dump_results()


        # filename, data = _read_data()
        # with open(filename, 'rb') as file:
        #     data = pickle.load(file)
        # data['success'] = False
        # with open(filename, 'wb') as file:
        #     data = pickle.dump(data, file, protocol=2)
