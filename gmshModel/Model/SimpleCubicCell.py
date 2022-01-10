################################################################################
#  CLASS FOR SIMPLE CUBIC UNIT CELL MESHES GENERATED USING THE GMSH-PYTHON-API #
################################################################################
# This file provides a class definition for a generation of unit cells with a
# simple cubic distribution of the inclusions. The class inherits from the
# GenericUnitCell class and extends it in order to specify the remaining
# placeholder methods of the GenericModel. Especially, methods to determine the
# cells size and place the inclusions are provided.

###########################
# Load required libraries #
###########################
# Standard Python libraries
import numpy as np                                                              # numpy for fast array computations
import copy as cp                                                               # copy  for deepcopies

# self defined class definitions and modules
from .GenericUnitCell import GenericUnitCell                                    # generic unit cell class definition (parent class)


################################
# Define SimpleCubicCell class #
################################
class SimpleCubicCell(GenericUnitCell):
    """Class definition for simple cubic unit cells

    This class provides required information for simple cubic unit cells. It
    inherits from the InclusionRVE class and extends its attributes and methods
    to handle the inclusion placement.

    The simple cubic unit cell allows to create "real" unit cells by passing the
    inclusion distance to the classes initialization method. If the cells size is
    specified instead, the distance is calculated: this allows for unit cells
    with a cuboid particle distribution

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
        size of the simple cubic unit cell (allow box-shaped cells)
        -> size=[Lx, Ly, (Lz)]

    origin: list/array
        origin of the simple cubic unit cell
        -> origin=[Ox, Oy, (Oz)]

    inclusionType: string
        string defining the type of inclusion
        -> iunclusionType= "Sphere"/"Cylinder"/"Circle"

    inclusionAxis:list/array
        array defining the inclusion axis (only relevant for inclusionType "Cylinder")
        -> currently restricted to Cylinders parallel to one of the coordinate axes
        -> inclusionAxes=[Ax, Ay, Az]

    relevantAxes: list/array
        array defining the relevant axes for distance calculations

    periodicityFlags: list/array
        flags indicating the periodic axes of the simple cubic unit cell
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
        """Initialization method for SimpleCubicCell object instances

        Parameters:
        -----------
        radius: float
            radius of the unit cells inclusions

        distance: float
            distance of the inclusions within the unit cell (for automatic size calculation)

        numberCells: list/array
            array providing the number of cells in the individual axis directions
            -> numberCells=[nx, ny, (nz)]

        size: list/array
            size of the simple cubic unit cell (allow box-shaped cells)
            -> size=[Lx, Ly, (Lz)]

        origin: list/array
            origin of the simple cubic unit cell
            -> origin=[Ox, Oy, (Oz)]

        inclusionType: string
            string defining the type of inclusion
            -> iunclusionType= "Sphere"/"Cylinder"/"Circle"

        inclusionAxis:list/array
            array defining the inclusion axis (only relevant for inclusionType "Cylinder")
            -> currently restricted to Cylinders parallel to one of the coordinate axes
            -> inclusionAxes=[Ax, Ay, Az]

        periodicityFlags: list/array
            flags indicating the periodic axes of the simple cubic unit cell
            -> periodicityFlags=[0/1, 0/1, 0/1]

        domainGroup: string
            name of the group the unit cells domain should belong to

        inclusionGroup: string
            name of the group the unit cells inclusions should belong to

        gmshConfigChanges: dict
            dictionary for user updates of the default Gmsh configuration
        """

        # initialize parents classes attributes and methods
        super().__init__(size=size,distance=distance,numberCells=numberCells,radius=radius,inclusionType=inclusionType,inclusionAxis=inclusionAxis,origin=origin,periodicityFlags=periodicityFlags,gmshConfigChanges=gmshConfigChanges)



################################################################################
#                 SPECIFIED/OVERWRITTEN PLACEHOLDER METHODS                    #
################################################################################

    ###############################################
    # Internal method to determine the cells size #
    ###############################################
    def _getCellSize(self,distance,inclusionType,inclusionAxis):

        # determine size of one unit cell
        if inclusionType == "Sphere":                                           # unit cell is three-dimensional with spherical inclusions
            unitSize=[distance, distance, distance]                             # -> define cell size to be equal to the inclusion distance in all directions
        elif inclusionType == "Cylinder":                                       # unit cell is three-dimensional with cylindrical inclusions
            cylinderAxis = np.array(np.nonzero(inclusionAxis)).flatten()        # -> get index of cylinder axis
            planeAxes=np.setdiff1d(np.array([0,1,2]),cylinderAxis)              # -> get indices of remaining axes
            unitSize=np.asarray(inclusionAxis).astype(float)                    # -> prepare size array (account for thickness in cylinder axis direction)
            unitSize[planeAxes]=[distance, distance]                            # -> set cell size for relevant in-plane axes
        elif inclusionType=="Circle":                                           # unit cell is two-dimensional
            unitSize=[distance, distance, 0]                                    # -> define cell size to be equal to the inclusion distance in the x-y-plane

        # return total size (multiply by number of cells per direction)
        return unitSize*self.numberCells


    #################################################
    # Method for a simple cubic inclusion placement #
    #################################################
    def placeInclusions(self):
        """Method to place inclusions for the simple cubic unit cell"""

        # get available information
        origin=self.origin                                                      # origin of unit cell
        size=self.size                                                          # (total) size of unit cell
        N=cp.deepcopy(self.numberCells)                                         # number of cells
        axesFlags=np.zeros((3,))                                                # creaty auxiliary flag variable to indicate axes which are relevant for inclusion center calculation
        axesFlags[self.relevantAxes]=1                                          # set axesFlags to 1 for the relevant axes

        # correct number of cells for non-spherical inclusions
        if self.inclusionType in ["Circle", "Cylinder"]:                        # inclusion type is "Cylinder" or "Cirlce"
            # ensure only 1 cell in the out-of-plane direction to avoid problems
            # with boolean operations, etc
            outOfPlaneAxis=np.setdiff1d(np.array([0,1,2]),self.relevantAxes)
            N[outOfPlaneAxis]=1

        # calculate required information
        offset=size*axesFlags/(2*N)                                             # offset (per direction) of inclusion centers from cell boundaries
        P0=origin+offset                                                        # coordinates of first inclusion
        P1=origin+size-offset                                                   # coordinates of last inclusion

        # get center coordinates of all inclusions
        C=np.mgrid[P0[0]:P1[0]:N[0]*1j,P0[1]:P1[1]:N[1]*1j,P0[2]:P1[2]:N[2]*1j].reshape(3,-1).T

        # save relevant results in randomInclusions object
        self.inclusionInfo=np.c_[C, self.radius*np.ones((np.shape(C)[0],1))]
