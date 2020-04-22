'''
Created on 2020-04-08 14:55:21
Last modified on 2020-04-22 16:06:23
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
                             PROPAGATED, LINEAR, ALL, DISPLACEMENT, ON,
                             AC_ON)

# standard library
import abc


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


class StaticRiksStep(Step):

    method_name = 'StaticRiksStep'

    def __init__(self, name, previous='Initial', model=None,
                 description='', nlgeom=OFF, adiabatic=OFF, maxLPF=None,
                 nodeOn=OFF, maximumDisplacement=0., dof=0, region=None,
                 timeIncrementationMethod=AUTOMATIC, maxNumInc=100,
                 totalArcLength=1., initialArcInc=None, minArcInc=None,
                 maxArcInc=None, matrixStorage=SOLVER_DEFAULT,
                 extrapolation=LINEAR, fullyPlastic='', noStop=OFF,
                 maintainAttributes=False, useLongTermSolution=OFF,
                 convertSDI=PROPAGATED):
        '''
        Parameters
        ----------
        description : str
            Step description.
        nlgeom : bool
            Whether to allow for geometric nonlinearity.
        adiabatic : bool
            Whether to perform an adiabatic stress analysis.
        maxLPF : float
            Maximum value of the load proportionality factor.
        nodeOn : bool
            Whether to monitor the finishing displacement value at a node.
        maximumDisplacement : float
            Value of the total displacement (or rotation) at the node node and
            degree of freedom that, if crossed during an increment, ends the
            step at the current increment. Only applicable when nodeOn=ON.
        dof : int
            Degree of freedom being monitored. Only applicable when nodeOn=ON
        region : abaqus region object
            Vertex at which the finishing displacement value is being monitored.
            Only applicable when nodeOn=ON.
        timeIncrementationMethod : abaqus constant
        maxNumInc : int
            Number of incrementations in a step.
        totalArcLength : float
            Total load proportionality factor associated with the load in this
            step.
        initialArcInc : float
            Initial load proportionality factor.
        minArcInc : float
            Minimum arc length increment allowed.
        maxArcInc : Maximum arc length increment allowed.
        matrixStorage : abaqus constant
            Type of matrix storage.
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
        convertSDI : abaqus constant
            Whether to force a new iteration if severe discontinuities occur
            during an iteration.

        Notes
        -----
        -for further informations see p49-128 of [1].
        '''
        # computations
        initialArcInc = totalArcLength if initialArcInc is None else initialArcInc
        minArcInc = min(initialArcInc, 1e-5 * totalArcLength) if minArcInc is None else minArcInc
        maxArcInc = totalArcLength if maxArcInc is None else maxArcInc
        # create arg dict
        self.args = {'description': description,
                     'nlgeom': nlgeom,
                     'adiabatic': adiabatic,
                     'maxLPF': maxLPF,
                     'nodeOn': nodeOn,
                     'maximumDisplacement': maximumDisplacement,
                     'dof': dof,
                     'timeIncrementationMethod': timeIncrementationMethod,
                     'maxNumInc': maxNumInc,
                     'totalArcLength': totalArcLength,
                     'initialArcInc': initialArcInc,
                     'minArcInc': minArcInc,
                     'maxArcInc': maxArcInc,
                     'matrixStorage': matrixStorage,
                     'extrapolation': extrapolation,
                     'fullyPlastic': fullyPlastic,
                     'noStop': noStop,
                     'maintainAttributes': maintainAttributes,
                     'useLongTermSolution': useLongTermSolution,
                     'convertSDI': convertSDI,
                     }
        if nodeOn is ON and region:
            self.args['region'] = region

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


class FrequencyStep(Step):

    method_name = 'FrequencyStep'

    def __init__(self, name, previous='Initial', model=None, eigensolver=LANCZOS,
                 numEigen=ALL, description='', shift=0., minEigen=None,
                 maxEigen=None, vectors=None, maxIterations=30, blockSize=DEFAULT,
                 maxBlocks=DEFAULT, normalization=DISPLACEMENT,
                 propertyEvaluationFrequency=None, projectDamping=ON,
                 acousticDamping=AC_ON, acousticRangeFactor=1.,
                 frictionDamping=OFF, matrixStorage=SOLVER_DEFAULT,
                 maintainAttributes=False, simLinearDynamics=OFF,
                 residualModes=OFF, substructureCutoffMultiplier=5.,
                 firstCutoffMultiplier=1.7, secondCutoffMultiplier=1.1,
                 residualModeRegion=None, residualModeDof=None,
                 limitSavedEigenVectorRegion=None):
        '''
        Parameters
        ----------
        eigensolver : abaqus constant
            Arguments ignored if eigenSolver!=LANCZOS: blockSize, maxBlocks,
            normalization, propertyEvaluationFrequency.
            Arguments ignored if eigenSolver!=LANCZOS or AMS: minEigen,
            maxEigen, acousticCoupling.
            Arguments ignored if eigenSolver!=AMS: projectDamping,
            acousticRangeFactor, substructureCutoffMultiplier,
            firstCutoffMultiplier, secondCutoffMultiplier, residualModeRegion,
            regionalModeDof, limitSavedEigenVectorRegion.
        numEigen : int or abaqus constant
            Number of eigenvalues to be estimated.
        description : str
            Step description.
        shift : float
            Shift point in cycles per time.
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
            Ignored if eigensolver!=LANCZOS.
        normalization : abaqus constant
            Method for normalizing eigenvectors.
        propertyEvaluationFrequency : float
            Frequency at which to evaluate frequency-dependent properties for
            viscoelasticity, springs, and dashpots during the eigenvalues
            extraction.
        projectDamping : bool
            Whether to include projection of viscous and structural damping
            operators during AMS eigenvalue extraction.
        acousticCoupling : abaqus constant
            Type of acoustic-structural coupling in models with acoustic
            and structural elements coupled using the *TIE option or in models
            with ASI-type elements.
        acousticRangeFactor : float
            Ratio of the maximum acoustic frequency to the maximum structural
            frequency.
        frictionDamping : bool
            Whether to add to the damping matrix contributions due to friction
            effects.
        matrixStorage : abaqus constant
            Type of matrix storage.
        maintainAttributes : bool
            Whether to retain attributes from an existing step with the same
            name.
        simLinearDynamics : bool
            Whether to activate the SIM-based linear dynamics procedure.
        residualModes : bool
            Whether to include residual modes from an immediately preceding
            Static, LinearPerturbation step.
        substructureCutoffMultiplier : float
            Cutoff frequency for substructure eigenproblems.
        firstCutoffMultiplier : float
            First cutoff frequency for a reduced eigenproblem.
        secondCutoffMultiplier : float
            Second cutoff freqency for a reduced eigenproblem.
        residualModeRegion : sequence of str
            Name of a region for which residual modes are requested.
        residualModeDof : sequence of int
            Degrees of freedom for which residual modes are requested.
        limitSavedEigenvectorRegion : region object
            Region for which eigenvectors should be saved.

        Notes
        -----
        -for further informations see p49-65 of [1].
        '''
        # computations
        vectors = min(2 * numEigen, numEigen * 8) if vectors is None else vectors
        # create arg dict
        self.args = {
            'eigensolver': eigensolver,
            'numEigen': numEigen,
            'description': description,
            'shift': shift,
            'minEigen': minEigen,
            'maxEigen': maxEigen,
            'vectors': vectors,
            'maxIterations': maxIterations,
            'blockSize': blockSize,
            'maxBlocks': maxBlocks,
            'normalization': normalization,
            'propertyEvaluationFrequency': propertyEvaluationFrequency,
            'projectDamping': projectDamping,
            'acousticDamping': acousticDamping,
            'acousticRangeFactor': acousticRangeFactor,
            'frictionDamping': frictionDamping,
            'matrixStorage': matrixStorage,
            'maintainAttributes': maintainAttributes,
            'simLinearDynamics': simLinearDynamics,
            'residualModes': residualModes,
            'substructureCutoffMultiplier': substructureCutoffMultiplier,
            'firstCutoffMultiplier': firstCutoffMultiplier,
            'secondCutoffMultiplier': secondCutoffMultiplier,
            'residualModeRegion': residualModeRegion,
            'residualModeDof': residualModeDof,
            'limitSavedEigenVectorRegion': limitSavedEigenVectorRegion}
        # initialize parent
        Step.__init__(self, name, previous, model=model)
