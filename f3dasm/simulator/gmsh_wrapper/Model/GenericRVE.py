################################################################################
#          CLASS FOR RVE MESHES GENERATED USING THE GMSH-PYTHON-API            #
################################################################################
# This file provides a class definition for an RVE generation using Python and
# Gmsh. The class inherits from the GenericModel class and extends it in order
# order to handle the problems that are connected with the generation of models
# with periodicity constraints.
#
# Currently, the class is restricted to RVEs with rectangular (2D)/ box-shaped
# (3D) domains (explicitly assumed within the setupPeriodicity() method).

###########################
# Load required libraries #
###########################
# Standard Python libraries
import numpy as np                                                              # numpy for array computations
import copy as cp                                                               # copy for deepcopies of arrays

# self-defined class definitions and modules
from .GenericModel import GenericModel                                          # generic model class definition (parent class)


###########################
# Define GenericRVE class #
###########################
class GenericRVE(GenericModel):
    """Generic class for RVEs created using the Gmsh-Python-API

    Based on the GenericModel class, this class provides extra attributes and
    methods that all box-shaped RVEs should have: the definition of size, origin
    and periodicityFlags as additional attributes facilitates an update of the
    parents class placeholder method setupPeriodicity().

    Attributes:
    -----------
    dimension: int
        dimension of the model instance

    size: list/array
        size of the box-shaped RVE model
        -> size=[Lx, Ly, (Lz)]

    origin: list/array
        origin of the box-shaped RVE model
        -> origin=[Ox, Oy, (Oz)]

    periodicityFlags: list/array
        flags indicating the periodic axes of the box-shaped RVE model
        -> periodicityFlags=[0/1, 0/1, 0/1]
    """

    #########################
    # Initialization method #
    #########################
    def __init__(self,size=None,origin=[0,0,0],periodicityFlags=[1,1,1],gmshConfigChanges={}):
        """Initialization method for box-shaped RVE models

        Parameters:
        -----------
        size: list/array
            size of the box-shaped RVE model
            -> size=[Lx, Ly, (Lz)]

        origin: list/array
            origin of the box-shaped RVE model
            -> origin=[Ox, Oy, (Oz)]

        periodicityFlags: list/array
            flags indicating the periodic axes of the box-shaped RVE model
            -> periodicityFlags=[0/1, 0/1, 0/1]

        gmshConfigChanges: dict
            dictionary for user updates of the default Gmsh configuration
        """
        # plausibility checks for input variables:
        for varName, varValue in {"size": size, "origin": origin, "periodicityFlags": periodicityFlags}.items():
            if varValue is None:                                                # check if variable has a value
                raise TypeError("Variable \"{}\" not set! Check your input data.".format(varName))
            elif len(np.shape(varValue)) > 1:                                   # check for right amount of array dimensions
                raise ValueError("Wrong amount of array dimensions for variable \"{}\"! For a cuboid RVE, the variable \"{}\" can only be one-dimensional. Check your input data.".format(varName))
            elif not len(varValue) in [2,3]:                                    # check for right amount of values
                raise ValueError("Wrong number of values for variable \"{}\"! For a cuboid RVE, the variable \"{}\" has to have 2 or 3 values. Check your input data.".format(varName))
            elif varName is "size" and np.count_nonzero(varValue) not in [2,3]: # check for right amount of non-zero values (size only)
                raise ValueError("Wrong number of non-zero values for variable \"{}\"! Only 2D/3D RVEs supported: the variable \"{}\" has to have 2 or 3 non-zero values. Check your input data.".format(varName))

        # type conversion for input arguments
        size=np.asarray(size)                                                   # type conversion for size to numpy array
        origin=np.asarray(origin)                                               # type conversion for origin to numpy array
        periodicityFlags=np.asarray(periodicityFlags)                           # type conversion for periodicityFlags to numpy array

        # get dimension of RVE and correct potentially two-dimensional arrays
        dimension=np.count_nonzero(size)                                        # dimension equals number non-zero sizes
        if len(size) != 3:                                                      # check if size is not a three-dimensional array
            size=np.r_[size,0]                                                  # -> append 0
        if len(origin) != 3:                                                    # check if origin is not a three-dimensional array
            newOrigin=np.zeros(3)                                               # -> create new three-dimensional array
            newOrigin[size != 0]=origin                                         # -> assign values of origin to non-zero dimensions of new array
            origin=newOrigin                                                    # -> overwrite origin with new array

        # initialize parent (GenericModel) class attributes and methods
        super().__init__(dimension=dimension,gmshConfigChanges=gmshConfigChanges)

        # initialize attributes that all instances of GenericRVE should have
        self.origin=origin                                                      # initialize unset RVE origin
        self.size=size                                                          # initialize unset RVE size
        self.periodicityFlags=periodicityFlags                                  # initialize unset periodic axes flags for the RVE



################################################################################
#                 SPECIFIED/OVERWRITTEN PLACEHOLDER METHODS                    #
################################################################################

    #########################################################
    # Method to set up periodicity constraints for the mesh #
    #########################################################
    def setupPeriodicity(self):
        """Overwritten method of the parent GenericModel class to define
        periodicity constraints for the model"""
        for iAx in range(0,self.dimension):                                     # loop over all axes
            if self.periodicityFlags[iAx] == 1:                                 # -> check if current axis is supposed to be periodic
                assocEnts,affineMatrix=self._getAssociatedEntities(iAx)         # ->-> get associated entities for the current axis
                self.gmshAPI.mesh.setPeriodic(self.dimension-1,assocEnts[1],assocEnts[0],affineMatrix) # ->-> set periodicity constraints



################################################################################
#           ADDITIONAL PRIVATE/HIDDEN METHODS FOR INTERNAL USE ONLY            #
################################################################################

    ###################################################################
    # Method to determine assosciated geometric entities of the model #
    ###################################################################
    def _getAssociatedEntities(self,axis):
        """Internal method to search for associated Gmsh entities on opposing
        sides of the RVE boundary"""
        # calculate required information from the RVE data
        bboxRVE=np.r_[[self.origin], [self.origin+self.size]]                   # bounding box of the RVE: bboxRVE=[[minBBoxPoint],[maxBBoxPoint]]
        tol=100*self.getGmshOption("Geometry.Tolerance")                        # tolerance for entity detection (factor 100 to find entities in wider bounding boxes)

        # calculate translation vector and affine transformation matrix
        transVec=np.zeros((3,1))                                                # initialize translation vector for the current pair of boundary entities
        transVec[axis]=bboxRVE[1,axis]-bboxRVE[0,axis]                          # update translation vector for the current pair of boundary entities
        affineMatrix=np.eye(4)                                                  # initialize affine transformation matrix for current pair of boundary entities
        affineMatrix[0:3,[-1]]=transVec                                         # update affine transformation matrix with translation information
        affineMatrix=affineMatrix.reshape(16).tolist()                          # convert affine transformation matrix to list for output

        # find associated entities for the current axes
        associatedEntities=[]                                                   # initialize list of associated entities
        for iSide in range(0,2):                                                # loop over both sides (positive/negative)
            bboxBnd=cp.deepcopy(bboxRVE)                                        # -> initialze bounding box for the current boundary as copy of bboxRVE
            bboxBnd[1-iSide,axis]=bboxRVE[iSide,axis]                           # -> modify the coordinate of the relevant dimension to match the boundary under consideration
            bboxBnd=bboxBnd+np.array([[-tol],[tol]])                            # -> add tolerances to the bounding box


            # find entities on boundary under consideration
            entityTags=self.getIDsFromTags(self.gmshAPI.getEntitiesInBoundingBox(*bboxBnd[0], *bboxBnd[1], self.dimension-1))
            associatedEntities.append(entityTags)

        # sort associated entities of current pair to prevent errors
        associatedEntities=self._sortAssociatedEntities(associatedEntities,transVec)

        # return associated entities and affine transformation matrix
        return associatedEntities, affineMatrix


    ###############################################################
    # Security function to sort entities in associated boundaries #
    ###############################################################
    def _sortAssociatedEntities(self,associatedEntities,translationVector):
        """Internal method to ensure a 1-1-mapping of associated Gmsh entities

        Parameters:
        -----------
        associatedEntities: list
            list of associated entities for the current pair of boundaries

        translationVector: array
            array defining the translation vector from one boundary to the other
        """
        # get required information from input arguments
        dim=self.dimension-1                                                    # entity dimension
        nEntities=len(associatedEntities[0])                                    # number of associated entities

        # convert associatedEntites to numpy array for calculations
        associatedEntities=np.asarray(associatedEntities)

        # get bounding boxes of all boundary entities on the negative side
        bboxesNeg=np.zeros((nEntities,6))
        for iEnt in range(0,nEntities):
            bboxesNeg[iEnt,:]=np.array(self.gmshAPI.getBoundingBox(dim,associatedEntities[0][iEnt]))

        # calculate associated boxes of corresponding entities on the positive side
        bboxesPos=bboxesNeg+np.c_[translationVector.T, translationVector.T]

        # search matching entities on positive side
        sortedIndices=np.zeros((nEntities),dtype=np.int8)
        for jEnt in range(0,nEntities):
            bboxEnt=np.array(self.gmshAPI.getBoundingBox(dim,associatedEntities[1][jEnt])) # find bounding box of current entity
            bboxesDiff=bboxesPos-bboxEnt                                        # calculate difference to all precomputed bounding boxes on positive side
            sortedIndices[jEnt]=np.argmin(np.linalg.norm(bboxesDiff,axis=1))       # determine index of entity with minimum distance of actual and precomputed bounding boxes

        # update sorting of entities on positive side
        associatedEntities[0]=associatedEntities[0][sortedIndices]

        # return updated associated entites
        return associatedEntities
