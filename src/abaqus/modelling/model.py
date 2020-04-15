'''
Created on 2020-04-08 14:29:12
Last modified on 2020-04-15 18:25:17
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


#%% object definition

# TODO: add fieldOutputs and historyOutputs

class BasicModel:

    def __init__(self, name, job_name, job_description):
        self.name = name
        self.job_name = job_name
        self.job_description = job_description
        # create model
        self.model = mdb.Model(name=self.name)
        backwardCompatibility.setValues(reportDeprecated=False)
        if 'Model-1' in mdb.models.keys():
            del mdb.models['Model-1']
        # initialize variables
        self.geometry_objects = []
        self.materials = []
        self.steps = ['Initial']
        self.bcs = []
        self.inp_additions = []

    def create_model(self):

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

    def write_inp(self):

        # create inp
        modelJob = mdb.Job(name=self.job_name, model=self.name,
                           description=self.job_description)
        modelJob.writeInput(consistencyChecking=OFF)

        # add lines to inp
        for inp_addition in self.inp_additions:
            inp_addition.write_text()

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

    # TODO: add possibility to dump model
