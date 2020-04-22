'''
Created on 2020-04-22 14:53:01
Last modified on 2020-04-22 20:38:49
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------


Notes
-----
-abaqus cae is kept open during simulation time (appropriate if running time
is low).
'''


#%% imports

# standard library
import pickle
import importlib

# local library
from .misc import convert_dict_unicode_str


#%% object definition

class RunModel(object):

    def __init__(self, abstract_model, data, sim_info):
        '''

        Notes
        -----
        -assumes the same data is required to instantiate each model of the
        sequence.

        '''
        self.abstract_model = self._import_abstract_model(abstract_model)
        self.data = data
        self.sim_info = sim_info
        self.models = []

    def run_models(self):

        for i, (model_name, info) in enumerate(self.sim_info.items()):

            # get args
            args = self.data.copy()
            args.update(info)

            # instantiate model
            previous_model = self.models[i - 1] if i else None
            model = self.abstract_model(name=model_name,
                                        previous_model=previous_model, **args)

            # create and run model
            model.create_model()
            model.write_inp(submit=True)

            # dump and save model
            model.dump(create_file=False)
            self.models.append(model)

        # TODO: make post-processing
        # TODO: dump results

    def _import_abstract_model(self, abstract_model):

        module_name, method_name = abstract_model.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, method_name)


if __name__ == '__main__':

    # read file
    # TODO: assume only one pickle file exists
    filename = 'simul.pkl'
    with open(filename, 'rb') as file:
        data = convert_dict_unicode_str(pickle.load(file))

    # create/load run model
    # TODO: verify if RunModel already exists
    run_model = RunModel(abstract_model=data['abstract_model'],
                         data=data['data'], sim_info=data['sim_info'])

    # run models
    run_model.run_models()
