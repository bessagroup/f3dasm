################################################################################
#      CLASS FOR FCC UNIT CELL MESHES GENERATED USING THE GMSH-PYTHON-API      #
################################################################################
# This file provides a class definition for a generation of unit cells with a
# face-centered cubic distribution of the inclusions. The class inherits from the
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


######################################
# Define FaceCenteredCubicCell class #
######################################
class FaceCenteredCubicCell(GenericUnitCell):
    """Class definition for face-centered cubic unit cells

    This class provides required information for face-centered cubic unit cells.
    It inherits from the GenericUnitCell class and extends its attributes and
    methods to handle the inclusion placement.

    The face-centered cubic unit cell allows to create "real" unit cells by
    passing the inclusion distance to the classes initialization method. If the
    cells size is specified instead, the distance is calculated: this allows for
    unit cells with a "fcc-like" particle distribution

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
        size of the face-centered cubic unit cell (allow box-shaped cells)
        -> size=[Lx, Ly, (Lz)]

    origin: list/array
        origin of the face-centered cubic unit cell
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
        flags indicating the periodic axes of the face-centered cubic unit cell
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
        """Initialization method for hexagonal unit cell object instances

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
            size of the face-centered cubic unit cell (allow box-shaped cells)
            -> size=[Lx, Ly, (Lz)]

        origin: list/array
            origin of the face-centered cubic unit cell
            -> origin=[Ox, Oy, (Oz)]

        inclusionType: string
            string defining the type of inclusion
            -> iunclusionType= "Sphere"/"Cylinder"/"Circle"

        inclusionAxis:list/array
            array defining the inclusion axis (only relevant for inclusionType "Cylinder")
            -> currently restricted to Cylinders parallel to one of the coordinate axes
            -> inclusionAxes=[Ax, Ay, Az]

        periodicityFlags: list/array
            flags indicating the periodic axes of the face-centered cubic unit cell
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
        if inclusionType == "Sphere":                                           # unit cell is three-dimensional with spherical inclusion
            unitSize=[distance, distance, distance]                             # -> define normal cell size for a face-centered cubic unit cell
        elif inclusionType == "Cylinder":                                       # unit cell is three-dimensional with cylindrical inclusion
            cylinderAxis = np.array(np.nonzero(inclusionAxis)).flatten()        # -> get index of cylinder axis
            planeAxes=np.setdiff1d(np.array([0,1,2]),cylinderAxis)              # -> get indices of remaining axes
            unitSize=np.asarray(inclusionAxis).astype(float)                    # -> prepare size array (account for thickness in cylinder axis direction)
            unitSize[planeAxes]=[2*distance, 2*distance]                        # -> set cell size for relevant in-plane axes
        elif inclusionType == "Circle":                                         # unit cell is two-dimensional with circular inclusion
            unitSize=[2*distance, 2*distance, 0]                                # -> define size of face-centered cubic cell in  x-y-plane

        # return total size (multiply by number of cells per direction)
        return unitSize*self.numberCells


    ##############################################
    # Method for a hexagonal inclusion placement #
    ##############################################
    def placeInclusions(self):
        """Method to place inclusions for the hexagonal unit cell"""

        # get available information
        origin=self.origin                                                      # origin of unit cell
        size=self.size                                                          # (total) size of unit cell
        N=cp.deepcopy(self.numberCells)                                         # number of cells
        step=size/N                                                             # step to get from one cell to another

        # determine offsets for inclusion placement using numpys mgrid
        # -> distinguish between spherical and non-spherical inclusions for
        #    plausible unit cells
        if self.inclusionType in ["Circle", "Cylinder"]:                        # inclusion type is "Cylinder" or "Cirlce"

            # determine indicator which axes are relevant
            firstAxesFlag=np.zeros(3)                                           # initialize indicator for first axis
            secondAxesFlag=np.zeros(3)                                          # initialize indicator for second axis
            firstAxesFlag[self.relevantAxes[0]]=1                               # set first relavant axis as active
            secondAxesFlag[self.relevantAxes[1]]=1                              # set second relavant axis as active

            # ensure only 1 cell in the out-of-plane direction to avoid problems
            # with boolean operations, etc
            outOfPlaneAxis=np.setdiff1d(np.array([0,1,2]),self.relevantAxes)
            N[outOfPlaneAxis]=1

            # set offsets for different sets of inclusions
            # -> divide by N to account for mutliple cells
            offsets=np.array([[0, 0, 0],                                        # -> corner inclusions
                              size/2*firstAxesFlag,                             # -> face-centered inclusions along first axis
                              size/2*secondAxesFlag])/N                         # -> face-centered inclusions along second axis
        else:                                                                   # inclusion type is "Sphere"
            # determine offsets for individual sets of inclusions
            # -> divide by N to account for mutliple cells
            offsets=np.array([[0, 0, 0],                                        # corner inclusions
                              [size[0]/2, size[1]/2, 0],                        # face-centered inclusions in bottom and top layers
                              [0, size[1]/2, size[2]/2],                        # inclusions of center layer with x=Lx/2
                              [size[0]/2, 0, size[2]/2]])/N                     # inclusions of center layer with x=0 and x=L

        # determine inclusion center points
        C=np.empty(shape=(0,3))                                                 # initialize empty array for center points
        for offset in offsets:                                                  # loop over all sets of inclusions
            P0=origin+offset                                                    # set starting point for point generation using mgrid
            P1=origin+size-step+offset                                          # set end point for point generation using mgrid
            n=cp.deepcopy(N)                                                    # copy number of cells (deepcopy to allow changes)
            for ax in self.relevantAxes:                                        # loop over all axes for inclusion placement
                if offset[ax]+self.radius > step[ax]:                           # offset is too big, i.e., inclusion leaves domain at the end
                    P0[ax]=origin[ax]+offset[ax]-step[ax]                       # -> adjust starting point for mesh generation with mgrid (incorporate periodic copy of inclusion that leaves the domain)
                    n[ax]+=1                                                    # -> increase number of repetitions by 1 to account for additional point
                elif offset[ax] < self.radius:                                  # offset is too low, i.e., inclusion leaves domain at the start
                    P1[ax]=origin[ax]+size[ax]+offset[ax]                       # -> adjust end point for mesh generation with mgrid (incorporate periodic copy of inclusion that leaves the domain)
                    n[ax]+=1                                                    # -> increase number of repetitions by 1 to account for additional point
            C=np.r_[C,np.mgrid[P0[0]:P1[0]:n[0]*1j,P0[1]:P1[1]:n[1]*1j,P0[2]:P1[2]:n[2]*1j].reshape(3,-1).T] # determine center points and append them to C

        # save relevant results in randomInclusions object
        self.inclusionInfo=np.c_[C, self.radius*np.ones((np.shape(C)[0],1))]
