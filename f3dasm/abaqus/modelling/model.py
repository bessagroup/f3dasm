'''
Created on 2020-04-08 14:29:12
Last modified on 2020-09-29 10:35:08
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Create general purpose models classes from which new models can inherit.

Notes
-----
-Modularized structure to allow easy overwrite of non-applicable methods.
'''


#%% imports

# abaqus
from abaqus import mdb, backwardCompatibility
from abaqusConstants import OFF
from abaqus import session

# standard library
import os
import pickle
import abc
import inspect


# object definition

class AbstractModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, job_info):
        self.name = name
        self.job_info = job_info

    @abc.abstractmethod
    def create_model(self):
        pass

    @abc.abstractmethod
    def write_inp(self, submit=True):
        pass

    @abc.abstractmethod
    def perform_post_processing(self, odb):
        pass

    def dump(self, create_file=True):

        # stop storing model
        self.model = None

        # create file
        if create_file:
            data = {'model': self}
            filename = '%s.pkl' % self.job_info['name']
            with open(filename, 'wb') as f:
                pickle.dump(data, f)


class BasicModel(AbstractModel):

    def __init__(self, name, job_info):
        '''
        Parameters
        ----------
        job_info: dict
            Must contain at least `name`.
        '''
        AbstractModel.__init__(self, name, job_info)
        # create model
        self.model = mdb.Model(name=self.name)
        backwardCompatibility.setValues(reportDeprecated=False)
        if 'Model-1' in mdb.models.keys():
            del mdb.models['Model-1']
        # initialize variables
        self.abort = None
        self.geometry_objects = []
        self.materials = []
        self.steps = ['Initial']
        self.bcs = []
        self.contact_properties = []
        self.interactions = []
        self.output_requests = []
        self.inp_additions = []

    def create_model(self):

        # assemble puzzle
        self.abort = self._assemble_puzzle()

        # if order to abort from `_assemble_puzzle`, then it stops
        if self.abort:
            return

        # create materials
        self._create_materials()

        # create parts
        self._create_parts()

        # create instances
        self._create_instances()

        # create steps
        self._create_steps()

        # create boundary conditions
        self._create_bcs()

        # create contact properties
        self._create_contact_properties()

        # create interactions
        self._create_interactions()

        # create outputs
        self._create_outputs()

    def write_inp(self, submit=False):

        # abort if model was not created
        if self.abort:
            return

        # create inp
        modelJob = mdb.Job(model=self.name, **self.job_info)
        modelJob.writeInput(consistencyChecking=OFF)

        # add lines to inp
        for inp_addition in self.inp_additions:
            inp_addition.write_text()

        # submit
        if submit:
            if len(self.inp_additions):
                filename = '%s.inp' % self.job_info['name']
                job_info = {key: value for key, value in self.job_info.items() if key != 'description'}
                modelJob = mdb.JobFromInputFile(inputFileName=filename,
                                                **job_info)
            modelJob.submit(consistencyChecking=OFF)
            modelJob.waitForCompletion()

    def _create_materials(self):
        for material in self.materials:
            material.create_material(self.model)

    def _create_parts(self):
        for obj in self.geometry_objects:
            obj.create_part(self.model)

    def _create_instances(self):
        for obj in self.geometry_objects:
            obj.create_instance(self.model)

    def _create_steps(self):
        for step in self.steps[1:]:
            step.create_step(self.model)

    def _create_bcs(self):
        for bc in self.bcs:
            bc.apply_bc(self.model)

    def _create_contact_properties(self):
        for contact_property in self.contact_properties:
            contact_property.create_contact_property(self.model)

    def _create_interactions(self):
        for interaction in self.interactions:
            interaction.create_interaction(self.model)

    def _create_outputs(self):

        # initialization
        create_field = False
        create_history = False

        # create requested field outputs
        for output in self.output_requests:
            output.create_output(self.model)
            if output.method_name == 'HistoryOutputRequest' and output.name != 'H-Output-1':
                create_history = True
            elif output.method_name == 'FieldOutputRequest' and output.name != 'F-Output-1':
                create_field = True

        # delete existing fields
        if create_history:
            del self.model.historyOutputRequests['H-Output-1']
        if create_field:
            del self.model.fieldOutputRequests['F-Output-1']

    def _update_list(self, variable, new_value):

        if type(new_value) is list or type(new_value) is tuple:
            variable.extend(new_value)
        else:
            variable.append(new_value)


class WrapperModel(AbstractModel):

    def __init__(self, name, job_info, abstract_model, post_processing_fnc=None,
                 previous_model=None, previous_model_results=None,
                 **kwargs):
        AbstractModel.__init__(self, name, job_info)
        self.abstract_model = abstract_model
        self.post_processing_fnc = post_processing_fnc
        self.previous_model = previous_model
        self.previous_model_results = previous_model_results
        self.kwargs = kwargs
        # deal with model and job names
        self.abstract_model_args = inspect.getargspec(abstract_model).args
        if 'job_name' in self.abstract_model_args:
            self.kwargs['job_name'] = self.job_info['name']
        if 'name' in self.abstract_model_args:
            self.kwargs['name'] = name
        elif 'model_name' in self.abstract_model_args:
            self.kwargs['model_name'] = name
        # avoid errors due to excessive arg definition
        for key, value in self.kwargs.items():
            if key not in self.abstract_model_args:
                del self.kwargs[key]
        # initialize variables
        self.abort = None

    def create_model(self):
        '''
        Notes
        -----
        * Results of previous models (cases for which there's consecutive
        simulations) can be passed with the argument `previous_model_results`.
        Additionally, you can pass the previous model job name using
        `previous_model_job_name`.
        '''
        # get previous model results
        if self.previous_model_results is None and self.previous_model:
            # access odb
            odb_name = '{}.odb'.format(self.previous_model.job_info['name'])
            odb = session.openOdb(name=odb_name)
            self.previous_model_results = self.previous_model.perform_post_processing(odb)
            odb.close()
            self.kwargs['previous_model_results'] = self.previous_model_results
            # verify if previous model job name should be passed
            if 'previous_model_job_name' in self.abstract_model_args:
                self.kwargs['previous_model_job_name'] = self.previous_model.job_info['name']

        # create model
        self.abort = self.abstract_model(**self.kwargs)

    def write_inp(self, submit=False):

        # abort if model was not created
        if self.abort:
            return

        # create inp
        if os.path.exists('{}.inp'.format(self.job_info['name'])):
            job_info = {key: value for key, value in self.job_info.items() if key != 'description'}
            modelJob = mdb.JobFromInputFile(
                inputFileName='{}.inp'.format(self.job_info['name']),
                **job_info)
        else:
            modelJob = mdb.Job(model=self.name, **self.job_info)
            modelJob.writeInput(consistencyChecking=OFF)

        # submit
        if submit:
            modelJob.submit(consistencyChecking=OFF)
            modelJob.waitForCompletion()

    def perform_post_processing(self, odb):

        return self.post_processing_fnc(odb)
