################################################################################
#     CLASS FOR INCLUSION RVE MESHES GENERATED USING THE GMSH-PYTHON-API       #
################################################################################
# This file provides a class definition for a generation of RVEs with randomly
# placed inclusions. The class inherits from the InclusionRVE class and extends
# it in order order to specify the remaining placeholder methods of the
# GenericModel. Methods to create the geometry, define refinement information
# and additional information for required boolean operations and physical groups
# are part of the class.

###########################
# Load required libraries #
###########################
# Standard Python libraries
import numpy as np                                                              # numpy for fast array computations
import copy as cp                                                               # copy  for deepcopies

# self defined class definitions and modules
from .InclusionRVE import InclusionRVE                                          # generic RVE class definition (parent class)


###################################
# Define RandomInclusionRVE class #
###################################
class RandomInclusionRVE(InclusionRVE):
    """Class definition for box-shaped RVEs with randomly distributed inclusions

    This class provides required information for box-shaped, RVEs with randomly
    distributed inclusions. It inherits from the InclusionRVE class and extends
    its attributes and methods to handle the inclusion placement as well as the
    definition of required boolean operations and physical groups.

    Attributes:
    -----------
    dimension: int
        dimension of the model instance

    inclusionSets: list/array
        array providing the necessary information for sets of inclusions to be
        placed
        -> inclusionSets=[radius, amount] (for the individual sets of inclusions)

    size: list/array
        size of the box-shaped RVE model
        -> size=[Lx, Ly, (Lz)]

    origin: list/array
        origin of the box-shaped RVE model
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
        flags indicating the periodic axes of the box-shaped RVE model
        -> periodicityFlags=[0/1, 0/1, 0/1]

    inclusionInfo: array
        array containing relevant inclusion information (center, radius) for
        distance calculations

    domainGroup: string
        name of the group the RVE domain should belong to

    inclusionGroup: string
        name of the group the inclusions should belong to

    gmshConfigChanges: dict
        dictionary for user updates of the default Gmsh configuration
    """
    #########################
    # Initialization method #
    #########################
    def __init__(self,inclusionSets=None,size=None,inclusionType=None,inclusionAxis=None,origin=[0,0,0],periodicityFlags=[1,1,1],domainGroup="domain",inclusionGroup="inclusions",gmshConfigChanges={}):
        """Initialization method for RandomInclusionRVE object instances

        Parameters:
        -----------
        inclusionSets: list/array
            array providing the necessary information for sets of inclusions to be
            placed
            -> inclusionSets=[radius, amount] (for the individual sets of inclusions)

        size: list/array
            size of the box-shaped RVE model
            -> size=[Lx, Ly, (Lz)]

        origin: list/array
            origin of the box-shaped RVE model
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
            flags indicating the periodic axes of the box-shaped RVE model
            -> periodicityFlags=[0/1, 0/1, 0/1]

        inclusionInfo: array
            array containing relevant inclusion information (center, radius) for
            distance calculations

        domainGroup: string
            name of the group the RVE domain should belong to

        inclusionGroup: string
            name of the group the inclusions should belong to

        gmshConfigChanges: dict
            dictionary for user updates of the default Gmsh configuration
        """
        # initialize parents classes attributes and methods
        super().__init__(size=size,inclusionType=inclusionType,inclusionAxis=inclusionAxis,origin=origin,periodicityFlags=periodicityFlags,gmshConfigChanges=gmshConfigChanges)

        # plausibility checks for input variables
        if inclusionSets is None:
            raise TypeError("Variable \"inclusionSets\" not set! For RVEs with random inclusion distributions, the inclusionSets must be defined. Check your input data.")

        # update inclusion sets to start placement with biggest inclusions
        inclusionSets=np.atleast_2d(inclusionSets)                              # type conversion for inclusionSets to be matrix-like
        inclusionSets=inclusionSets[inclusionSets[:,0].argsort(axis=0)[::-1]]   # sort inclusionSets (descending) to start algorithm with biggest inclusions
        self.inclusionSets=inclusionSets                                        # save updated inclusionSets array to class object

        # set attributes from input parameters
        self.domainGroup=domainGroup                                            # set group name for the domain object
        self.inclusionGroup=inclusionGroup                                      # set group name for the inclusion objects

        # set default placement options
        self.placementOptions={
            "maxAttempts": 10000,                                               # maximum number of attempts to place one inclusion
            "minRelDistBnd": 0.1,                                               # minimum relative (to inclusion radius) distance to the domain boundaries
            "minRelDistInc": 0.1,                                               # minimum relative (to inclusion radius) distance to other inclusions
        }



################################################################################
#                 SPECIFIED/OVERWRITTEN PLACEHOLDER METHODS                    #
################################################################################

    ############################################################################
    # Method to define the required geometric objects for the model generation #
    ############################################################################
    def defineGeometricObjects(self,placementOptions={}):
        """Overwritten method of the GenericModel class to define and create the
        required geometric objects for the model generation

        Parameters:
        -----------
        placementOptions: dict
            dictionary for user updates of the default placement options
        """
        # update placement options
        placementOptions=self.updatePlacementOptions(placementOptions)

        # generate geometry
        self.addGeometricObject(self.domainType,group=self.domainGroup,origin=self.origin,size=self.size) # add domain object to calling RVE
        self.placeInclusions(placementOptions)                                  # call internal inclusion placement routine
        for i in range(0,np.shape(self.inclusionInfo)[0]):                      # loop over all inclusions
            if self.inclusionType in ["Sphere","Circle"]:
                self.addGeometricObject(self.inclusionType,group=self.inclusionGroup,center=self.inclusionInfo[i,0:3],radius=self.inclusionInfo[i,3]) # add inclusions to calling RVE object
            elif self.inclusionType == "Cylinder":
                self.addGeometricObject(self.inclusionType,group=self.inclusionGroup,center=self.inclusionInfo[i,0:3],radius=self.inclusionInfo[i,3],axis=self.inclusionAxis) # add inclusions to calling RVE object


    ###################################################
    # Method for the definition of boolean operations #
    ###################################################
    def defineBooleanOperations(self):
        """Overwritten method of the GenericModel class to define the required
        boolean operations for the model generation

        Normally, the definition of basic geometric objects is not sufficient
        to generate the RVE geometry. To this end, boolean operations can be
        used to generate more complex RVEs from the basic geometric objects.
        The required boolean operations for the generation of an RVE with
        randomly distributed inclusions is defined here.
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
    def definePhysicalGroups(self):
        """Overwritten method of the GenericModel class to define and the
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
#               ADDITIONAL METHODS FOR THE INCLUSION PLACEMENT                 #
################################################################################

    ##########################################
    # Method to get update placement options #
    ##########################################
    def updatePlacementOptions(self,optionsUpdate):
        """Method to updated the inclusion placement options

        Parameters:
        -----------
        optionsUpdate: dict
            dictionary storing the updates for the currently set placement options
        """
        self.placementOptions.update(optionsUpdate)
        return self.placementOptions


    #########################################
    # Method for random inclusion placement #
    #########################################
    def placeInclusions(self,placementOptions):
        """Method to place inclusions for the RVE geometry

        Within this method, the inclusions for the user-defined inclusion sets
        (sets of inclusion radii and amounts) are placed in a - up to now -
        standard random-close-packing algorithm. This allows for moderate volume
        fractions of inclusions with identical radii. For high volume fractions,
        the algorithm needs to be refined. The applied method ensures periodicity
        in the user-defined directions and allows to define minimum distances
        between two inclusions as well as the inclusion surface and the surrounding
        domain boundaries (to facilitate meshing).

        Parameters:
        -----------
        placementOptions: dict
            dictionary with options for the inclusion placement
        """
        # get inlusion radii and amounts for the individual sets
        rSets=np.asarray(self.inclusionSets[:,0])                               # radii of inclusion for the individual sets
        nSets=self.inclusionSets[:,1]                                           # amount of inclusions for the individual sets
        distBndSets=rSets*placementOptions["minRelDistBnd"]                     # distance of inclusions to the domain boundaries for the individual sets
        distIncSets=rSets*placementOptions["minRelDistInc"]                     # distance of inclusions to other inclusions for the individual sets

        # initialization
        totalInstancesSet=0                                                     # number of already set inclusion instances (inclusions and copies of them)
        incInfo=np.zeros((4*np.sum(nSets).astype(int),6))                       # incInfo=[ [incCoords], rInc, mindDistBnd, mindDistInc] -> assume every inclusion to cut potentially two boundaries (i.e. to have three copies) for initialization
        self.placementInfo=np.zeros(np.shape(rSets))                            # information about placed inclusions for the individual sets

        # loop over all incSets
        relevantAxesFlags=np.zeros((3,))                                        # creaty auxiliary flag variable to indicate axes which are relevant for inclusion center calculation
        relevantAxesFlags[self.relevantAxes]=1                                  # set relevantAxesFlags to 1 for the relevant axes
        for iSet in range(0,np.shape(rSets)[0]):

            # initialization:
            nAttempts=0                                                         # attempts to place the current inclusion
            placedIncsForSet=0                                                  # amount of placed inclusions for the current set

            # try to place all inclusions for the current set
            while placedIncsForSet < nSets[iSet] and nAttempts < placementOptions["maxAttempts"]:

                # calculate random coordinates for current inclusion (ensure 2D array for proper handling)
                thisIncInfo=np.atleast_2d(np.r_[np.random.rand(3)*self.size*relevantAxesFlags, rSets[iSet], distBndSets[iSet], distIncSets[iSet]])

                # check inclusion distance to boundaries
                acceptInc, distBnd = self._checkBndDistance(thisIncInfo[0,:],self.relevantAxes)
                if acceptInc == False:                                          # start with next attempt, if inclusion is not accepted
                    nAttempts+=1
                    continue
                else:                                                           # set number of required inclusion instances to 1, if inclusion is accepted
                    thisIncInstances=1

                # generate periodic copies, if necessary
                if any(self.periodicityFlags) == True:
                    thisIncInfo, thisIncInstances = self._periodicIncs(thisIncInfo,distBnd)

                # check inclusion distance to inclusions placed so far
                acceptInc=True                                                  # initially assume that current inclusion is accepted
                if totalInstancesSet > 0:                                       # only check if at least one inclusion has been placed
                    for i in range(0,thisIncInstances):                         # check i-th instance of current inclusion
                        acceptIncInstance = self._checkIncDistance(thisIncInfo[i,:],incInfo[0:totalInstancesSet,:],self.relevantAxes) # check if i-th instance is accepted
                        if acceptIncInstance == False:                          # do not check other instances, if this one is not accepted
                            acceptInc=False                                     # do not accept current inclusion, if one of the instances is not accepted
                            break

                    # check if current inclusion is accepted or not
                    if acceptInc == False:                                      # start with next attempt, if inclusion is not accepted
                        nAttempts+=1
                        if nAttempts == placementOptions["maxAttempts"]:        # user info, if not all inclusions of the current set could be placed
                            print("Could not place all inclusions for the current set. For r={0:.2f}, {1:.0f}/{2:.0f} have been placed.".format(rSets[iSet],placedIncsForSet,nSets[iSet]))
                        continue
                    else:                                                       # update incInfo and number of set inclusions/inclusion instances
                        incInfo[totalInstancesSet:totalInstancesSet+thisIncInstances,:]=thisIncInfo
                        placedIncsForSet+=1
                        totalInstancesSet+=thisIncInstances
                else:                                                           # set first inclusion
                    incInfo[totalInstancesSet:totalInstancesSet+thisIncInstances,:]=thisIncInfo
                    placedIncsForSet+=1
                    totalInstancesSet+=thisIncInstances

                # save number of placed inclusions for the current set to placementInfo
                self.placementInfo[iSet]=placedIncsForSet

        # prepare incInfo for output (get relevant rows and cols)
        incInfo[:,0:3]=incInfo[:,0:3]+np.atleast_2d(self.origin)                # translate center coordinates to match with defined origin of the RVE domain
        incInfo=incInfo[0:totalInstancesSet,0:4]                                # incInfo=[M_x, M_y, M_z, R] with M: center points, R: radii

        # save relevant results in randomInclusions object
        self.inclusionInfo=incInfo


    #######################################################
    # Method to save inclusion information to a text file #
    #######################################################
    def saveIncInfo(self,file):
        """Method to save inclusion information to delimited ascii file

        Parameters:
        -----------
        file: string
            name of the file to save the inclusion information in
        """
        fileDir,fileName,fileExt=self._getFileParts(file,"Misc")                # split fileName into file parts
        with open(fileDir+"/"+fileName+fileExt,"w") as incInfoFile:             # open file
            np.savetxt(incInfoFile,self.inclusionInfo)                          # save information to file



################################################################################
#           ADDITIONAL PRIVATE/HIDDEN METHODS FOR INTERNAL USE ONLY            #
################################################################################

    ####################################################
    # Method to generate periodic copies of inclusions #
    ####################################################
    def _periodicIncs(self,thisIncInfo,distBnd):
        """Internal method to generate periodic copies for inclusions cutting
        the domain boundaries

        Parameters:
        -----------
        thsiIncInfo: array
            array containing information of the inclusion that cuts the boundary

        distBnd: array
            distance of the inclusion to the RVE boundary
        """
        # check if inclusion cuts the boundaries and get indices
        #
        # bndCuts variable:
        #------------------
        # 1st row of "distBnd < 0" shows cuts with negative x,y,z-surfaces (boolean)
        # 2nd row of "distBnd < 0" shows cuts with positive x,y,z-surfaces (boolean)
        # -> 1st row of "bndCuts" defines if negative (0) or positive (1) boundary are cut -> sign of necessary displacement of original inclusion
        # -> 2nd row of "bndCuts" defines direction for which the cut has been found -> direction for which displacement has to be performed
        bndCuts=np.array(np.where(distBnd < 0))

        # loop over all directions for which copies have to be generated
        thisIncInstances=1                                                      # initialize number of required inclusion instances
        for i in range(0,np.shape(bndCuts)[1]):                                 # find all direction for which copies have to be generated
            if self.periodicityFlags[bndCuts[1,i]]==1:                               # only create periodic copy if current direction is marked as periodic
                thisIncCopies=cp.deepcopy(thisIncInfo)                          # initialize current copy with current inclusion data (deepcopy)
                thisIncCopies[:,bndCuts[1,i]]=thisIncCopies[:,bndCuts[1,i]]+(-1)**bndCuts[0,i]*self.size[bndCuts[1,i]] # modify inclusion centers of copies
                thisIncInfo=np.r_[thisIncInfo,thisIncCopies]                    # append copied inclusions to current inclusion data
                thisIncInstances=np.shape(thisIncInfo)[0]                       # update number of required inclusion instances

        # return relevant data
        return thisIncInfo, thisIncInstances
