'''
Created on 2020-04-08 14:55:21
Last modified on 2020-04-08 15:51:11
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Define abaqus steps.

References
----------
1. Simulia (2015). ABAQUS 2016: Scripting Reference Guide
'''


#%% imports

# abaqus
from abaqusConstants import (LANCZOS, DEFAULT, SOLVER_DEFAULT)

# standard library
import abc


#%% abstract classes

class Step(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, previous):
        '''
        Parameters
        ----------
        name : str
        previous : str
            Previous step
        '''
        self.name = name
        self.previous = previous

    def create_step(self, model):

        # get method
        create_step = getattr(model, self.method_name)

        # create step
        create_step(self.name, self.previous, **self.args)


#%% particular step definition

class BuckleStep(Step):

    method_name = 'BuckleStep'

    def __init__(self, name, previous='Initial', numEigen=20, description='',
                 eigensolver=LANCZOS, minEigen=None, maxEigen=None,
                 vectors=None, maxIterations=30, blockSize=DEFAULT,
                 maxBlocks=DEFAULT, matrixStorage=SOLVER_DEFAULT,
                 maintainAttributes=False):
        '''
        Parameters
        ----------
        numEigen : int
            Number of eigenvalues to be estimated.
        description : str
            Step description.
        eigensolver : abaqus constant
        minEigen : float
            Minimum eigenvalue of interest.
        maxEigen : float
            Maximum eigenvalue of interest.
        vectors : int
            Number of vectors used in each iteration.
        maxIterations : int
            Maximum number of iterations.
        blockSize : abaqus constant or int
            Size of the Lanczos block steps.
        maxBlocks : abaqus constant or int
            Maximum number of Lanczos block steps within each Lanczos run.
        matrixStorage : abaqus constant
            Type of matrix storage
        maintainAttributes : bool
            Whether to retain attributes from an existing step with the same
            name.

        Notes
        -----
        -for further informations see p49-10 of [1].
        -minEigen, blockSize and maxBlocks are ignored if eigensolver!=LANCZOS.
        '''
        # initialize parent
        Step.__init__(self, name, previous)
        # computations
        vectors = min(2 * numEigen, numEigen * 8) if vectors is None else vectors

        # create arg dict
        self.args = {'numEigen': numEigen,
                     'description': description,
                     'eigensolver': eigensolver,
                     'minEigen': minEigen,
                     'maxEigen': maxEigen,
                     'vectors': vectors,
                     'maxIterations': maxIterations,
                     'blockSize': blockSize,
                     'maxBlocks': maxBlocks,
                     'matrixStorage': matrixStorage,
                     'maintainAttributes': maintainAttributes}
