'''
Created on 2020-04-08 14:55:21
Last modified on 2020-04-09 17:02:14
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
from abaqusConstants import (LANCZOS, DEFAULT, SOLVER_DEFAULT, OFF,
                             AUTOMATIC, NONE, DIRECT, RAMP, FULL_NEWTON,
                             PROPAGATED, LINEAR)

# standard library
import abc


# TODO: bcs should not be part of a step, but contain step information somehow.


#%% abstract classes

class Step(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, previous, model=None):
        '''
        Parameters
        ----------
        name : str
        previous : str
            Previous step
        model : abaqus mdb object
        '''
        self.name = name
        self.previous = previous
        # computations
        if model:
            self.create_step(model)

    def create_step(self, model):

        # get method
        create_step = getattr(model, self.method_name)

        # create step
        create_step(name=self.name, previous=self.previous, **self.args)


#%% particular step definition

class StaticStep(Step):

    method_name = 'StaticStep'

    def __init__(self, name, previous='Initial', model=None, description='',
                 timePeriod=1., nlgeom=OFF, stabilizationMethod=NONE,
                 stabilizationMagnitude=2e-4, adiabatic=OFF,
                 timeIncrementationMethod=AUTOMATIC, maxNumInc=100,
                 initialInc=None, minInc=None, maxInc=None,
                 matrixSolver=DIRECT, matrixStorage=SOLVER_DEFAULT,
                 amplitude=RAMP, extrapolation=LINEAR, fullyPlastic='',
                 noStop=OFF, maintainAttributes=False,
                 useLongTermSolution=OFF, solutionTechnique=FULL_NEWTON,
                 reformKernel=8, convertSDI=PROPAGATED,
                 adaptiveDampingRatio=0.05, continueDampingFactors=OFF):
        '''
        Parameters
        ----------
        description : str
            Step description.
        timePeriod : float
            Total time period.
        nlgeom : bool
            Whether to allow for geometric nonlinearity.
        stabilizationMethod : abaqus constant
            Stabilization type.
        stabilizationMagnitude : float
            Damping intensity of the automatic damping algorithm. Ignored if
            stabilizationMethod=None.
        adiabatic : bool
            Whether to perform an adiabatic stress analysis.
        timeIncrementationMethod : abaqus constant
        maxNumInc : int
            Number of incrementations in a step.
        initalInc : float
            Initial time increment.
        minInc : float
            Minimum tome increment allowed.
        maxInc : float
            Maximum time increment allowed.
        matrixSolver : abaqus constant
            Type of solver.
        matrixStorage : abaqus constant
            Type of matrix storage.
        amplitude : abaqus constant
            Amplitude variation for loading magnitudes during the step.
        extrapolation : abaqus constant
            Type of extrapolation to use in determining the incremental solution
            for a nonlinear analysis.
        fullyPlastic : str
            Region being monitored for fully plastic behavior.
        noStop : bool
            Whether to accept the solution to an increment after the maximum
            number of interations allowed has been completed, even if the
            equilibrium tolerances are not satisfied.
        maintainAttributes : bool
            Whether to retain attributes from an existing step with the same
            name.
        useLongTermSolution : bool
            Whether to obtain the fully relaxed long-term elastic solution with
            time-domain viscoelasticity or the long-term elastic-plastic solution
            for two-layer viscoplasticity.
        solutionTechnique : abaqus constant
            Technique used for solving nonlinear equations.
        reformKernel : int
            Number of quasi-Newton iterations allowed before the kernel matrix
            is reformed.
        convertSDI : abaqus constant
            Whether to force a new iteration if severe discontinuities occur
            during an iteration.
        adaptiveDampingRatio : float
            Maximum allowable ratio of the stabilization energy to the total
            strain energy. Ignored if stabilizationMethod=None.
        continueDampingFactors : bool
            Whether this step will carry over the damping factors from the
            results of the preceding general step.

        Notes
        -----
        -for further informations see p49-134 of [1].
        '''
        # computations
        initialInc = timePeriod if initialInc is None else initialInc
        minInc = min(initialInc, timePeriod * 1e-5) if minInc is None else minInc
        maxInc = timePeriod if maxInc is None else maxInc
        # create args dict
        self.args = {'description': description,
                     'timePeriod': timePeriod,
                     'nlgeom': nlgeom,
                     'stabilizationMethod': stabilizationMethod,
                     'stabilizationMagnitude': stabilizationMagnitude,
                     'adiabatic': adiabatic,
                     'timeIncrementationMethod': timeIncrementationMethod,
                     'maxNumInc': maxNumInc,
                     'initialInc': initialInc,
                     'minInc': minInc,
                     'maxInc': maxInc,
                     'matrixSolver': matrixSolver,
                     'matrixStorage': matrixStorage,
                     'amplitude': amplitude,
                     'extrapolation': extrapolation,
                     'fullyPlastic': fullyPlastic,
                     'noStop': noStop,
                     'maintainAttributes': maintainAttributes,
                     'useLongTermSolution': useLongTermSolution,
                     'solutionTechnique': solutionTechnique,
                     'reformKernel': reformKernel,
                     'convertSDI': convertSDI,
                     'adaptiveDampingRatio': adaptiveDampingRatio,
                     'continueDampingFactors': continueDampingFactors}
        # initialize parent
        Step.__init__(self, name, previous, model=model)


class BuckleStep(Step):

    method_name = 'BuckleStep'

    def __init__(self, name, previous='Initial', model=None, numEigen=20,
                 description='', eigensolver=LANCZOS, minEigen=None,
                 maxEigen=None, vectors=None, maxIterations=30, blockSize=DEFAULT,
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
            Minimum eigenvalue of interest. Ignored if eigensolver!=LANCZOS.
        maxEigen : float
            Maximum eigenvalue of interest.
        vectors : int
            Number of vectors used in each iteration.
        maxIterations : int
            Maximum number of iterations.
        blockSize : abaqus constant or int
            Size of the Lanczos block steps. Ignored if eigensolver!=LANCZOS.
        maxBlocks : abaqus constant or int
            Maximum number of Lanczos block steps within each Lanczos run.
            Ignored if eigensolver!=LANCZOS.
        matrixStorage : abaqus constant
            Type of matrix storage.
        maintainAttributes : bool
            Whether to retain attributes from an existing step with the same
            name.

        Notes
        -----
        -for further informations see p49-10 of [1].
        '''
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
        # initialize parent
        Step.__init__(self, name, previous, model=model)
