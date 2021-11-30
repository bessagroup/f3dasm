################################################################################
#CLASS FOR INCLUSION-BASED UNIT CELL MESHES GENERATED USING THE GMSH-PYTHON-API#
################################################################################
# This file provides a class definition for a generation of inclusion-based unit
# cells. The class inherits from the InclusionRVE class and extends it in order
# to specify the requirements of all derived unit cells child classes.

###########################
# Load required libraries #
###########################
# Standard Python libraries
import numpy as np                                                              # numpy for fast array computations
import copy as cp                                                               # copy  for deepcopies

# self defined class definitions and modules
from .InclusionRVE import InclusionRVE                                          # generic RVE class definition (parent class)


################################
# Define GenericUnitCell class #
################################
class GenericUnitCell(InclusionRVE):
    """Class definition for unit cells

    This class provides required information for inclusion-based unit cells. It
    inherits from the InclusionRVE class and extends its attributes and methods
    to handle the boolean operations and the definition of physical groups.

    All unit cell allow to create "real" unit cells by passing the inclusion
    distance to the classes initialization method. If the cells size is
    specified instead, the distance is calculated automatically: this allows for
    unit cells with an inclusion distribution that is close to physical unit
    cells but gives more freedom in their generation.

    Attributes:
    -----------
    dimension: int
        dimension of the model instance

    distance: float
        distance of the inclusions within the unit cell (for automatic size calculation)

    radius: float
        radius of the unit cells inclusions

    numberCells: list/array
        array providing the number of cells in the individual axis directions
        -> numberCells=[nx, ny, (nz)]

    size: list/array
        size of the unit cell (allow box-shaped cells)
        -> size=[Lx, Ly, (Lz)]

    origin: list/array
        origin of the unit cell
        -> origin=[Ox, Oy, (Oz)]

    inclusionType: string
        string defining the type of inclusion
        -> iunclusionType= "Sphere"/"Cylinder"/"Circle"

    inclusionAxis:list/array
        array defining the inclusion axis (only relevant for inclusionType "Cylinder")
        -> currently restricted to Cylinders parallel to one of the coordinate axes
        -> inclusionAxis=[Ax, Ay, Az]

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
    def __init__(self,distance=None,radius=None,numberCells=[1,1,1],size=None,inclusionType=None,inclusionAxis=None,origin=[0,0,0],periodicityFlags=[1,1,1],domainGroup="domain",inclusionGroup="inclusions",gmshConfigChanges={}):
        """Initialization method for GenericUnitCell object instances

        Parameters:
        -----------
        distance: float
            distance of the inclusions within the unit cell
            -> for automatic size calculation

        radius: float
            radius of the unit cells inclusions

        numberCells: list/array
            array providing the number of cells in the individual axis directions
            -> numberCells=[nx, ny, (nz)]

        size: list/array
            size of the unit cell (allow box-shaped cells)
            -> for automatic distance calculation
            -> size=[Lx, Ly, (Lz)]

        origin: list/array
            origin of the unit cell
            -> origin=[Ox, Oy, (Oz)]

        inclusionType: string
            string defining the type of inclusion
            -> iunclusionType= "Sphere"/"Cylinder"/"Circle"

        inclusionAxis:list/array
            array defining the inclusion axis (only relevant for inclusionType "Cylinder")
            -> currently restricted to Cylinders parallel to one of the coordinate axes
            -> inclusionAxis=[Ax, Ay, Az]

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
        # check if size or distance are passed, throw exception if none or both are passed
        if distance is None and size is None:
            raise TypeError("Neither the unit cells size, nor the inclusion distance are passed. One of both arguments has to be specified to allow for a proper inclusion placement. Check your input.")
        if distance is not None and size is not None:
            raise TypeError("Duplicate information for the unit cell detected. To prevent conflicting information, only one of the variables \"distance\" and \"size\" is supposed to be passed. Check your input.")

        # additionally check if inclusionType is passed, since it is needed here
        # (This is a duplicate check, since inclusionType will be checked again
        #  within the parent classes __init__)
        if inclusionType is None:
            raise TypeError("No inclusion type passed. To generated an inclusion-based unit cell, the type of inclusions must be specified. Check your input.")
        if inclusionType == "Cylinder" and inclusionAxis is None:
            raise TypeError("No cylinder axis passed. For inclusions of type cylinder, the length and direction must be specified with the variable \"inclusionAxis\". Check your input")

        # plausibility checks for remaining input variables
        if radius is None:                                                      # check if radius has a value
            raise TypeError("Variable \"radius\" not set! For an inclusion-based unit cell, the radius must be defined. Check your input data.")
        if numberCells is None:                                                 # check if numberCells has a value
            raise TypeError("Variable \"numberCells\" not set! Check your input data.")
        elif len(np.shape(numberCells)) > 1:                                    # check for right amount of array dimensions
            raise ValueError("Wrong amount of array dimensions for variable \"numberCells\"! For an inclusion-based unit cell, the variable \"numberCells\" can only be one-dimensional. Check your input data.")
        elif not len(numberCells) in [2,3]:                                     # check for right amount of values
            raise ValueError("Wrong number of values for variable \"numberCells\"! For an inclusion-based unit cell, the variable \"numberCells\" has to have 2 or 3 values. Check your input data.")
        elif np.any(np.asarray(numberCells)==0):                                # check that only non-zero numberCells are given
            raise ValueError("Detected non-zero number of cells in variable \"numberCells\"! In an inclusion-based unit cell, the number of cells in all axis directions must be non-zero. Check your input data.")

        # set attributes from input parameters
        self.radius=radius                                                      # save inclusion radius
        self.numberCells=np.asarray(numberCells)                                # save number of cells per axis direction
        self.domainGroup=domainGroup                                            # set group name for the domain object
        self.inclusionGroup=inclusionGroup                                      # set group name for the inclusion objects

        # determine size, if inclusion distance is given
        if size is None:
            size=self._getCellSize(distance,inclusionType,inclusionAxis)

        # initialize parent classes attributes and methods
        super().__init__(size=size,inclusionType=inclusionType,inclusionAxis=inclusionAxis,origin=origin,periodicityFlags=periodicityFlags,gmshConfigChanges=gmshConfigChanges)



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
            if self.inclusionType in ["Sphere","Circle"]:
                self.addGeometricObject(self.inclusionType,group=self.inclusionGroup,center=self.inclusionInfo[i,0:3],radius=self.inclusionInfo[i,3]) # add inclusions to calling RVE object
            elif self.inclusionType == "Cylinder":
                self.addGeometricObject(self.inclusionType,group=self.inclusionGroup,center=self.inclusionInfo[i,0:3],radius=self.inclusionInfo[i,3],axis=self.inclusionAxis*self.numberCells) # add inclusions to calling RVE object (multiply by number of cells to ensure correct cylinder length)


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
#             PLACEHOLDER METHODS TO BE SPECIFIED IN CHILD CLASSES             #
################################################################################

    ##########################################################
    # Internal method for the cell-specific size calculation #
    ##########################################################
    def _getCellSize(self,distance,inclusionType,inclusionAxis):
        """Placeholder methof to determine the cell size of an inclusion-based
        unit cell"""
        pass

    ####################################################
    # Method for the cell-specific inclusion placement #
    ####################################################
    def placeInclusions(self):
        """Placeholder method to place inclusions for the inclusion-based unit cell"""
        pass
