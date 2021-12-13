from abaqus import session
import pickle

import traceback
from abaqus import mdb
from abaqusConstants import OFF

from ..utils import convert_dict_unicode_str

class RunModelFromInp(object):

    def __init__(self):
        self.filename, data = _read_data()
        self.job_info = data['config']
        #model_name = self.sim_info['name']

    def execute(self):
        filename = '%s.inp' % self.job_info['name']
        job_info = {key: value for key, value in self.job_info.items() if key != 'description'}
        modelJob = mdb.JobFromInputFile(inputFileName=filename,
                                        **job_info)
        modelJob.submit(consistencyChecking=OFF)
        modelJob.waitForCompletion()

def _read_data():
    filename = 'sim_config.pkl'
    # read file
    with open(filename, 'rb') as file:
        data = convert_dict_unicode_str(pickle.load(file))
    return filename, data


if __name__ == '__main__':

    try: 
        runjob = RunModelFromInp()
        runjob.execute()
    except:
        # create error file
        with open('sim_error.txt', 'w') as file:
            traceback.print_exc(file=file)

        # update success flag
        filename, data = _read_data()
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        data['success'] = False
        with open(filename, 'wb') as file:
            data = pickle.dump(data, file, protocol=2)
