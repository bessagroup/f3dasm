################################################################################
# CLASS FOR INCLUSION-BASED HELICAL CHAINS GENERATED USING THE GMSH-PYTHON-API #
################################################################################
# This file provides a class definition for a generation of helical chains for
# circular and spherical inclusions. The class inherits from the InclusionRVE
# class and extends it in order to specify the requirements of meshes for
# helical chains.

###########################
# Load required libraries #
###########################
# Standard Python libraries
import numpy as np                                                              # numpy for fast array computations
import copy as cp                                                               # copy  for deepcopies

# self defined class definitions and modules
from .InclusionRVE import InclusionRVE                                          # InclusionRVE class definition (parent class)


################################
# Define GenericUnitCell class #
################################
class HelicalChain(InclusionRVE):
    """Class definition for helical chains (circular and spherical inclusions only)

    This class provides required information for meshes of unit cells with
    helical chains. It inherits from the InclusionRVE class and extends its
    attributes and methods to handle the geometry generation, boolean operations
    and the definition of physical groups.

    The HelicalChain class currently supports spherical and circular inclusions
    which are arranged in a chain that is parallel to one of the coordinate
    axes. The slope of the chain is determined from the passed size of the chain,
    the angle theta between neighboring inclusions (pi for circular inclusions,
    since it is the only reasonable choice) and the chain axis. One unit cell is
    supposed to contain one full helix.

    Attributes:
    -----------
    dimension: int
        dimension of the model instance

    chainRadius: float
        radius of the helical chain

    inclusionRadius: float
        radius of the helical chain inclusions

    theta: float
        angle (radian) between two inclusions of the helical chain

    chainDirection: list/array
        array defining the chain axis direction
        -> currently restricted to chains parallel to one of the coordinate axes

    numberCells: list/array
        array providing the number of cells in the individual axis directions
        -> numberCells=[nx, ny, (nz)]

    size: list/array
        size of the helical chains surrounding unit cell (box-shaped cell)
        -> if the angle alpha is passed, only the sizes perpendicular to the chain axis direction have to be passed
        -> size=[Lx, Ly, (Lz)]

    origin: list/array
        origin of the unit cell
        -> origin=[Ox, Oy, (Oz)]

    inclusionType: string
        string defining the type of inclusion
        -> inclusionType= "Sphere"/"Circle"

    relevantAxes: list/array
        array defining the relevant axes for distance calculations

    periodicityFlags: list/array
        flags indicating the periodic axes of the unit cell
        -> periodicityFlags=[0/1, 0/1, 0/1]

    inclusionInfo: array
        array containing relevant inclusion information (center, radius) for
        distance calculations

    domainGroup: string
        name of the group the unit cells domain should belong to

    inclusionGroup: string
        name of the group the unit cells inclusions should belong to

    gmshConfigChanges: dict
        dictionary for user updates of the default Gmsh configuration
    """
    #########################
    # Initialization method #
    #########################
    def __init__(self,chainRadius=None,theta=None,inclusionRadius=None,numberCells=[1,1,1],size=None,inclusionType=None,chainDirection=[0,0,1],origin=[0,0,0],periodicityFlags=[1,1,1],domainGroup="domain",inclusionGroup="inclusions",gmshConfigChanges={}):
        """Initialization method for HelicalChain object instances

        Parameters:
        -----------
        chainRadius: float
            radius of the helical chain

        inclusionRadius: float
            radius of the helical chain inclusions

        theta: float
            angle (radian) between two inclusions of the helical chain

        chainDirection: list/array
            array defining the chain axis direction
            -> currently restricted to chains parallel to one of the coordinate axes
            -> chainDirection=[0/1, 0/1, (0/1)]

        numberCells: list/array
            array providing the number of cells in the individual axis directions
            -> for two-dimensional problems, nz is automatically set to 1
            -> numberCells=[nx, ny, (nz)]

        size: list/array
            size of the helical chains surrounding unit cell (box-shaped cell)
            -> size=[Lx, Ly, (Lz)]

        origin: list/array
            origin of the unit cell
            -> origin=[Ox, Oy, (Oz)]

        inclusionType: string
            string defining the type of inclusion
            -> inclusionType= "Sphere"/"Circle"

        periodicityFlags: list/array
            flags indicating the periodic axes of the unit cell
            -> periodicityFlags=[0/1, 0/1, 0/1]

        domainGroup: string
            name of the group the unit cells domain should belong to

        inclusionGroup: string
            name of the group the unit cells inclusions should belong to

        gmshConfigChanges: dict
            dictionary for user updates of the default Gmsh configuration
        """
        # initialize parent classes attributes and methods
        super().__init__(size=size,inclusionType=inclusionType,origin=origin,periodicityFlags=periodicityFlags,gmshConfigChanges=gmshConfigChanges)

        # check inclusionType for additional ristrictions
        if inclusionType == "Cylinder":                                         # throw exception for cylinders (not supported)
            raise ValueError("No support for cylindrical inclusions in unit cells of helical chains. Check your input")
        elif inclusionType == "Circle":                                         # circular inclusion
            theta=np.pi                                                         # -> set angle theta to 180 degrees (only plausible possibility)

        # check if non-calculable scalar arguments are passed
        for varName, varValue in {"theta": theta, "chainRadius": chainRadius, "inclusionRadius": inclusionRadius}.items():
            if varValue is None:                                                # check if varValue is not set
                raise TypeError("Variable \"{}\" not set. For unit cells of a helical chain, the variable \"{}\" has to be passed. Check your input".format(varName))

        # check chain axis direction
        if chainDirection == None:                                              # chainDirection not passed
            raise TypeError("Variable \"chainDirection\" not passed. For unit cells of a helical chain, the chain axis direction must be passed. Check your input.")
        elif len(np.shape(chainDirection)) > 1:                                 # check for right amount of array dimensions
            raise ValueError("Wrong amount of array dimensions for variable \"{}\"! For unit cells of a helical chain, the variable \"{}\" can only be one-dimensional. Check your input data.".format(varName))
        elif not len(chainDirection) in [2,3]:                                  # check for right amount of values
            raise ValueError("Wrong number of values for variable \"{}\"! For unit cells of a helical chain, the variable \"{}\" has to have 2 or 3 values. Check your input data.".format(varName))
        elif np.count_nonzero(chainDirection) > 1:                              # chainDirection not parallel to one of the coordinate axis
            raise ValueError("Unsupported \"chainDirection\" passed. For units cell of a helical chain, only chainDirections which are parallel to one of the spatial axes are supported. Check your input.")
        chainDirection=np.asarray(chainDirection)                               # convert to numpy array

        # plausibility checks for number of cells
        if numberCells is None:                                                 # check if numberCells has a value
            raise TypeError("Variable \"numberCells\" not set! Check your input data.")
        elif len(np.shape(numberCells)) > 1:                                    # check for right amount of array dimensions
            raise ValueError("Wrong amount of array dimensions for variable \"numberCells\"! For a unit cell of a helical chain, the variable \"numberCells\" can only be one-dimensional. Check your input data.")
        elif not len(numberCells) in [2,3]:                                     # check for right amount of values
            raise ValueError("Wrong number of values for variable \"numberCells\"! For a unit cell of a helical chain, the variable \"numberCells\" has to have 2 or 3 values. Check your input data.")
        elif np.any(np.asarray(numberCells)==0):                                # check that only non-zero numberCells are given
            raise ValueError("Detected non-zero number of cells in variable \"numberCells\"! In a unit cell of a helical chain, the number of cells in all axis directions must be non-zero. Check your input data.")
        if self.dimension == 2:                                                 # ignore number of cells different from 1 for two-dimensional problems
            numberCells[2]=1
        numberCells=np.asarray(numberCells)                                     # convert to numpy array

        # plausibility check for size
        # -> ensure that chain fits into the surrounding cell since, otherwise,
        # -> the chain would not be a chain anymore and the placement would be
        # -> more involved
        chainDirInd=np.asarray(np.nonzero(chainDirection)).flatten()
        for axis in range(0,self.dimension):
            if axis != chainDirInd:
                assert self.size[axis]/numberCells[axis]-2*(chainRadius+inclusionRadius) > 0, "Helical chain does not fit into unit cell. To really create a unit cell of a helical chain, the chains are not allowed to overlap. Check your input."
            else:
                assert self.size[axis]/numberCells[axis]-2*inclusionRadius > 0, "Unreasonable choice of unit cells size in chain direction. In order to create a unit cell of a helical chain, at least one inclusion has to fit in chain axis direction. Check yout input."

        # set attributes from input parameters
        self.chainRadius=chainRadius                                            # save chain radius
        self.inclusionRadius=inclusionRadius                                    # save inclusion radius
        self.theta=theta                                                        # save angle theta
        self.chainDirection=chainDirection                                      # save chain axis direction
        self.numberCells=np.asarray(numberCells)                                # save number of cells per axis direction
        self.domainGroup=domainGroup                                            # set group name for the domain object
        self.inclusionGroup=inclusionGroup                                      # set group name for the inclusion objects


################################################################################
#                 SPECIFIED/OVERWRITTEN PLACEHOLDER METHODS                    #
################################################################################

    ############################################################################
    # Method to define the required geometric objects for the model generation #
    ############################################################################
    def defineGeometricObjects(self):
        """Overwritten method of the GenericModel class to define and create the
        required geometric objects for the model generation
        """

        # generate geometry
        self.addGeometricObject(self.domainType,group=self.domainGroup,origin=self.origin,size=self.size) # add domain object to calling RVE
        self.placeInclusions()                                                  # call internal inclusion placement routine
        for i in range(0,np.shape(self.inclusionInfo)[0]):                      # loop over all inclusions
            self.addGeometricObject(self.inclusionType,group=self.inclusionGroup,center=self.inclusionInfo[i,0:3],radius=self.inclusionInfo[i,3]) # add inclusions to calling RVE object


    ###################################################
    # Method for the definition of boolean operations #
    ###################################################
    def defineBooleanOperations(self):
        """Overwritten method of the GenericModel class to define the required
        boolean operations for the model generation
        """
        self.booleanOperations=[{                                               # first boolean operation (intersect domain group and inclusions group to get rid of parts that exceed the domain boundary)
            "operation": "intersect",                                           # -> intersection of "domain" and "inclusions"
            "object": self.domainGroup,                                         # -> use "domain" group as object
            "tool": self.inclusionGroup,                                        # -> use "inclusions" group as tool
            "removeObject": False,                                              # -> keep the object ("domain") after the boolean operation for further use
            "removeTool": True,                                                 # -> remove the tool ("inclusions") after the boolean operation
            "resultingGroup": self.inclusionGroup                               # -> assign the result of the boolean operation to the "inclusions" group (i.e. overwrite the group) for further use
        },
        {                                                                       # second boolean operation (cut domain with resulting inclusions to create holes within the domain where the inclusions are placed)
            "operation": "cut",                                                 # -> cut "inclusions" (updated group) from "domain"
            "object": self.domainGroup,                                         # -> use "domain" group as object
            "tool": self.inclusionGroup,                                        # -> use "inclusions" group as tool
            "removeObject": True,                                               # -> remove the object ("domain") after the boolean operation
            "removeTool": False,                                                # -> keep the tool ("inclusions") after the boolean operation for further use
            "resultingGroup": self.domainGroup                                  # -> assign the result of the boolean operation to the "domain" group (i.e. overwrite the group)
        }]


    ################################################
    # Method for the definition of physical groups #
    ################################################
    def definePhysicalGroups(self,**args):
        """Overwritten method of the GenericModel class to define the
        required physical groups for the model mesh generation

        In order to be able to assign different material properties to different
        regions in the generated mesh, physical groups are used in Gmsh to
        combine different regions into one group. The additional definition of
        a boundary group allows to identify the boundary of the mesh without
        searching for elements on the boundary within the solver.
        """
        # append boundary group to save boundary information in the mesh, too
        self.groups.update({"boundary": self.getBoundaryEntities()})

        # define physical groups
        self.physicalGroups=[{
            "dimension": self.dimension,                                        # define dimension of the physical group
            "group": self.domainGroup,                                          # define name of the physical group
            "physicalNumber": 1                                                 # set physical number of the physical group
        },
        {
            "dimension": self.dimension,                                        # define dimension of the physical group
            "group": self.inclusionGroup,                                       # define name of the physical group
            "physicalNumber": 2                                                 # set physical number of the physical group
        },
        {
            "dimension": self.dimension-1,                                      # define dimension of the physical group
            "group": "boundary",                                                # define name of the physical group
            "physicalNumber": 3,                                                # set physical number of the physical group
        }]



################################################################################
#                            INCLUSION PLACEMENT                               #
################################################################################

    ####################################################
    # Method for the cell-specific inclusion placement #
    ####################################################
    def placeInclusions(self):
        """Method to place the inclusions within a unit cell of a helical chain"""

        # get relevant information for inclusion placement
        theta=self.theta                                                        # helical angle between neigboring inclusions of the chain
        radius=self.inclusionRadius                                             # radius of the inclusions to place
        chainRadius=self.chainRadius                                            # radius of the chain
        chainDir=self.chainDirection                                            # chain axis direction
        numCells=self.numberCells                                               # number of cells to create
        size=self.size                                                          # size of the model
        origin=self.origin                                                      # origin of the model

        # get indices of chain and perpendicular axes
        chainAxis=np.asarray(np.nonzero(chainDir)).flatten()                    # index of the chain axis
        perpAxes=np.setdiff1d(np.arange(0,3),chainAxis)                         # indices of perpendicular axes

        # calculate information on chains and number of particles
        incsPerChain=(numCells[chainAxis]*(int(2*np.pi/theta))+1).item()        # get number of inclusions (per chain) in chain axis direction (account for repitition of first inclusion in top layer)
        numberChains=np.prod(numCells[perpAxes])                                # get number of chains in direction perpendicular to the chain
        totalIncs=numberChains*incsPerChain                                     # get total number of inclusions to place
        cellIndices=np.array(np.meshgrid(np.arange(0,numCells[perpAxes[0]]),np.arange(0,numCells[perpAxes[1]]))).T.reshape(-1,2) # determine indices of the different cells (i.e. chains) to allow for an easy placement

        # get required distances
        chainDist=np.zeros(3)                                                   # initialize distance between chains in directions perpendicular to the chain
        chainDist[perpAxes]=(size[perpAxes]).astype(float)/numCells[perpAxes]   # calculated distance between chains in directions perpendicular to the chain
        layerDist=size[chainAxis]/(incsPerChain-1)                              # get layer distance in chain axis direction

        # determine center points of all inclusions in all chains
        incInfo=np.zeros((totalIncs,4))                                         # initialize inclusion info array (center points and radii)
        currentInc=0                                                            # initialize counter for placed inclusions
        for iChain in range(0,numberChains):                                    # loop over all chains
            chainOrigin=origin+0.5*chainDist                                    # determine origin of the current chains axis
            chainOrigin[perpAxes]+=cellIndices[iChain,:]*chainDist[perpAxes]    # -> account for current chains/cells position
            helicalAngle=0                                                      # initialize helical angle
            for incInChain in range(0,incsPerChain):                            # loop over all inclusions of the current chain
                incInfo[currentInc,perpAxes]=chainOrigin[perpAxes]+chainRadius*np.asarray([np.cos(helicalAngle), np.sin(helicalAngle)]) # set coordinates perpendicular to chain direction
                incInfo[currentInc,chainAxis]=chainOrigin[chainAxis]+incInChain*layerDist # set height of the current inclusion
                incInfo[currentInc,-1]=radius                                   # add radius
                helicalAngle+=theta                                             # update helical angle
                currentInc+=1                                                   # update counter for placed inclusions

        # save inclusion information
        self.inclusionInfo=incInfo
