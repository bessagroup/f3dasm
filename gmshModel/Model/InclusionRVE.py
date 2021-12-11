################################################################################
#     CLASS FOR INCLUSION RVE MESHES GENERATED USING THE GMSH-PYTHON-API       #
################################################################################
# This file provides a class definition for a generation of RVEs with inclusions
# using Python and Gmsh. The class inherits from the GenericRVE class and extends
# it in order order to handle distance and refinement calculations
#
# Currently, the class is restricted to RVEs with rectangular (2D)/ box-shaped
# (3D) domains (explicitly assumed within the setupPeriodicity() method) which
# comprise inclusions that are all of the same type (explicitly assumed by using
# one inclusionInformation array and one inclusionAxis variable).
###########################
# Load required libraries #
###########################
# Standard Python libraries
import numpy as np                                                              # numpy for array computations
import copy as cp                                                               # copy for deepcopies of arrays

# self-defined class definitions and modules
from .GenericRVE import GenericRVE                                              # generic RVE class definition (parent class)


###########################
# Define GenericRVE class #
###########################
class InclusionRVE(GenericRVE):
    """Generic class for RVEs with inclusions created using the Gmsh-Python-API

    Based on the GenericRVE class, this class provides extra attributes and
    methods that all box-shaped RVEs with inclusions should have: the definition
    of an inclusion information array and relevant inclusion axes allows to
    provide additional methods for distance and refinement field calculations.

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
        flags indicating the periodic axes of the box-shaped RVE model
        -> periodicityFlags=[0/1, 0/1, 0/1]

    inclusionInfo: array
        array containing relevant inclusion information (center, radius) for
        distance calculations

    gmshConfigChanges: dict
        dictionary for user updates of the default Gmsh configuration
    """
    #########################
    # Initialization method #
    #########################
    def __init__(self,size=None,inclusionType=None,inclusionAxis=None,origin=[0,0,0],periodicityFlags=[1,1,1],gmshConfigChanges={}):
        """Initialization method for InclusionRVE objects

        Parameters:
        -----------
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

        periodicityFlags: list/array
            flags indicating the periodic axes of the box-shaped RVE model
            -> periodicityFlags=[0/1, 0/1, 0/1]

        gmshConfigChanges: dict
            dictionary for user updates of the default Gmsh configuration
        """

        # initialize parents classes attributes and methods
        super().__init__(size=size,origin=origin,periodicityFlags=periodicityFlags,gmshConfigChanges=gmshConfigChanges)

        # plausibility checks for additional input variables
        if inclusionType is None:
            raise TypeError("Variable \"inclusionType\" not set! For RVEs with inclusions, the type of inclusions must be defined. Check your input data.")
        elif inclusionType == "Cylinder":
            if inclusionAxis is None:
                raise TypeError("Inclusion type \"Cylinder\" specified but variable \"inclusionAxis\" not set! For RVEs with cylindrical inclusions, the cylinder axis must be defined. Check your input data.")
            elif len(inclusionAxis) != 3 or np.count_nonzero(inclusionAxis) != 1:
                raise ValueError("Wrong amount of non-zero elements in \"inclusionAxis\"! For cylindrical inclusions, the variable \"inclusionAxis\" has to specify the length and direction of the cylinder axis which, at the moment, has to be parallel to one of the coordinate axes. Check your input.")

        # determine relevant axes for distance calculations
        self.inclusionAxis=np.asarray(inclusionAxis)                            # save inclusion axis as numpy array to class object
        axesIndices=np.arange(0,3)                                              # array with indices of all axes
        if inclusionType == "Sphere":                                           # spherical inclusions
            self.relevantAxes=axesIndices                                       # -> all axes are relevant for distance
        elif inclusionType == "Cylinder":                                       # cylindrical inclusions with axis orientation parallel to one of the coordinate axes
            self.relevantAxes=axesIndices[self.inclusionAxis[:]==0]             # -> relevant axes are the ones perpendicular to the cylinder axis
        elif inclusionType == "Circle":                                         # circular/ disk-shaped inclusions
            self.relevantAxes=axesIndices[[0,1]]                                # -> always assume placement in x-y-plane

        # determine class names for domain and inclusions geometric objects
        self.inclusionType=inclusionType
        if self.dimension == 3:
            self.domainType="Box"
        elif self.dimension == 2:
            self.domainType="Rectangle"

        # define default refinement information for setRefinementInformation()
        self.refinementOptions={
            "maxMeshSize": "auto",                                              # automatically calculate maximum mesh size with built-in method
            "inclusionRefinement": True,                                        # flag to indicate active refinement of inclusions
            "interInclusionRefinement": True,                                   # flag to indicate active refinement of space between inclusions (inter-inclusion refinement)
            "elementsPerCircumference": 18,                                     # use 18 elements per inclusion circumference for inclusion refinement
            "elementsBetweenInclusions": 3,                                     # ensure 3 elements between close inclusions for inter-inclusion refinement
            "inclusionRefinementWidth": 3,                                      # use a relative (to inclusion radius) refinement width of 1 for inclusion refinement
            "transitionElements": "auto",                                       # automatically calculate number of transitioning elements (elements in which tanh function jumps from h_min to h_max) for inter-inclusion refinement
            "aspectRatio": 1.5                                                  # aspect ratio for inter-inclusion refinement: ratio of refinement in inclusion distance and perpendicular directions
        }

        # initialize required additional attributes for all inclusionRVEs
        self.inclusionInfo=[]                                                   # initialize unset inclusion information array



################################################################################
#                 SPECIFIED/OVERWRITTEN PLACEHOLDER METHODS                    #
################################################################################

    ####################################################
    # Method for automated refinementfield calculation #
    ####################################################
    def defineRefinementFields(self,refinementOptions={}):
        """Method to calculate refinement information for the RVE

        For inclusion-based RVEs, the inclusionInfo array can be used to
        calculate refinement fields based on inclusion radii and distances.
        These calculations are defined here so that every child class of
        InclusionRVE can use them to define  refinement fields.

        Parameters:
        -----------
        refinementOptions: dict
            user-defined updates for the default refinement options
        """

        # load default options and update them with passed user options
        self.updateRefinementOptions(refinementOptions)

        # restrict the maximum mesh size (set domain mesh size)
        self._setMathEvalField("const",self.refinementOptions["maxMeshSize"])

        # perform inclusion refinements with corresponding methods
        incInfo=self.getInclusionInfoForRefinement()                            # get extended inclusion information array with inclusions copied over close boundaries
        if self.refinementOptions["inclusionRefinement"] == True:               # refinement of inclusions is active
            self.inclusionRefinement(incInfo)                                   # -> perform refinement of inclusions and their boundaries (ensure set number of elements per circumference)
        if self.refinementOptions["interInclusionRefinement"] == True:          # refinement between different inclusions is active
            self.interInclusionRefinement(incInfo)                              # -> perform refinement between inclusions

        # merge all fields within one "Min"-Field
        relevantFields=np.arange(1,len(self.refinementFields)+1)                # start with "1" since Gmsh starts counting with 1
        self.refinementFields.append({"fieldType": "Min", "fieldInfos": {"FieldsList": relevantFields}})

        # set "Min"-Field as background field in Gmsh
        self.backgroundField=len(relevantFields)+1



################################################################################
#          ADDITIONAL METHODS FOR REFINEMENT INFORMATION CALCULATION           #
################################################################################

    #######################################
    # Method to update refinement options #
    #######################################
    def updateRefinementOptions(self,optionsUpdate):
        """Method to update refinement options

        Parameters:
        -----------
        optionsUpdate: dict
            dictionary containing user updates of the set refinement options
        """
        self.refinementOptions.update(optionsUpdate)                            # update refinement options
        if self.refinementOptions["maxMeshSize"] is "auto":                     # check if "maxMeshSize" is set to "auto"
            self.refinementOptions["maxMeshSize"]=self._calculateMaxMeshSize()  # -> calculate "maxMeshSize" with internal function


    ################################################################
    # Method to get an extended inclusionInfo array for refinement #
    ################################################################
    def getInclusionInfoForRefinement(self,relDistBnd=2):
        """Method to calculate an "extended" inclusionInfo for the refinement
        methods

        In order to ensure a periodicity of not only the geometry but also the
        mesh, the fields defined in the refinement methods, have to be periodic,
        i.e. present on both sides of periodic boundaries. Within this method,
        inclusions that are close to the domain boundaries are copied and stored
        in an "extended" inclusionInfo array that is only used within the
        refinement methods. This ensures that refinement fields that are found
        on one boundary will also be present on its periodic counterpart.

        Parameters:
        -----------
        relDistBnd: int/float
            distance (relative to inclusion radius) for which inclusion is considered
            to be "far" from the boundary if it is exceeded
        """

        # get relevant data for calculations
        incInfo=cp.deepcopy(self.inclusionInfo)                                 # get information of set inclusions (deepcopy to prevent changes of the model)
        incInfo[:,0:3]=incInfo[:,0:3]-np.atleast_2d(self.origin)                # temporariliy eliminate origin offset for all inclusions to simplify calculations
        bndPoints=np.asarray([[0, 0, 0], self.size])                            # temporariliy assume an RVE with origin at [0,0,0] to simplify calculations
        axes=self.relevantAxes                                                  # get relevant axes for distance calculations

        # initialize required arrays
        extIncInfo=np.zeros((27*np.shape(incInfo)[0],np.shape(incInfo)[1]))     # initialize extended incInfo array (ensure that all inclusion information will fit; get rid of unused space at the end of the function)
        totalIncInstances=0                                                     # initialize number of set inclusion instances

        # loop over all "original" inclusions
        for iInc in range(0,np.shape(incInfo)[0]):

            # get current inclusion from original incInfo array and initialize number of instances
            thisIncInfo=np.atleast_2d(incInfo[iInc,:])                          # information of original inclusion
            copyDirs=np.zeros((1,3),dtype=bool)                                 # array to mark copies of the original inclusion in specific directions
            thisIncInstances=1                                                  # number of instances for this inclusion

            # get distance of inclusion to boundaries
            distBndsCenter=np.absolute(self._getDistanceVector(thisIncInfo[0,:],bndPoints,axes)) # calculate (per-direction) distance of inclusion center to domain boundaries
            distBnds=distBndsCenter-thisIncInfo[0,3]                                             # get corresponding distance of inclusion boundary
            closeBnds=np.array(np.where((distBnds<=relDistBnd*thisIncInfo[0,3]) & (distBnds>0))) # check which inclusions are close to which boundaries of the domain; omit inclusions with negative distances to the boundaries, since they are periodic copies of other inclusions and do not need to be checked separately

            # loop over all "close" boundaries
            for iBnd in range(0,np.shape(closeBnds)[1]):
                validIncs=~copyDirs[:,axes[closeBnds[1,iBnd]]]                  # get valid inclusions to copy, i.e.: inclusions that have not been copied along this axis before
                thisIncCopies=cp.deepcopy(thisIncInfo[validIncs,:])             # initialize current copy with current inclusion data (deepcopy)
                isCopyInDir=cp.deepcopy(copyDirs[validIncs,:])                  # initialize indicator for copies in specific directions
                thisIncCopies[:,axes[closeBnds[1,iBnd]]]+=(-1)**closeBnds[0,iBnd]*self.size[axes[closeBnds[1,iBnd]]] # modify inclusion centers of copies (only if they are not already a copy in this direction -> prevent superfluous copies)
                isCopyInDir[:,axes[closeBnds[1,iBnd]]]=True                     # set current axis direction to True (indicate that this already is a copy in the specified direction)
                thisIncInfo=np.r_[thisIncInfo,thisIncCopies]                    # append copied inclusions to current inclusion data
                copyDirs=np.r_[copyDirs,isCopyInDir]                            # append markers to overall array

            # update extended incInfo array
            thisIncInstances=np.shape(thisIncInfo)[0]                           # number of instances for this inclusion
            extIncInfo[totalIncInstances:totalIncInstances+thisIncInstances,:]=thisIncInfo # save information on current inclusion and its copies
            totalIncInstances+=thisIncInstances                                 # updated number of total inclusion instances

        # prepare output arrays
        extIncInfo=extIncInfo[0:totalIncInstances,:]                            # get relevant colums of extIncinfo array
        extIncInfo[:,0:3]+=np.atleast_2d(self.origin)                           # add origin to extIncInfo array coordinates
        return extIncInfo


    ###################################################################
    # Method to perform refinement of inclusions and their boundaries #
    ###################################################################
    def inclusionRefinement(self,incInfo):
        """Method to perform refinement of inclusions and their boundaries

        Within this method, the inclusions are refined using a function similar
        to the normal distribution. This method ensures that especially the
        inclusion boundaries are refined whereas the inclusion centers and the
        surrounding matrix material generally remain coarse. The applied
        refinement function of type "gaussian" is described in the function
        definition of "_gaussianRefinement()".

        Parameters:
        -----------
        incInfo: array
            extended inclusionInfo array containing information on inclusions
            within the RVE model as well as outside but close to the model
            boundaries
        """
        # get required refinement options
        refinementOptions=self.refinementOptions
        elementsPerCircumference=refinementOptions["elementsPerCircumference"]
        inclusionRefinementWidth=refinementOptions["inclusionRefinementWidth"]
        maxMeshSize=refinementOptions["maxMeshSize"]

        # set refinement fields in loop over all inclusions
        for iInc in range(0,np.shape(incInfo)[0]):
            meshSize=2*np.pi*incInfo[iInc,3]/elementsPerCircumference           # get mesh size by dividing inclusion circumference by elementsPerCircumference
            refinementWidth=inclusionRefinementWidth*incInfo[iInc,3]            # determine refinementWidth
            sigma=refinementWidth/4                                             # determine standard deviation of gaussian function so that 95% of the area under the refinement function are within the given refinementWidth: sigma=refinementWidth/4
            self._setMathEvalField("gaussian",np.r_[incInfo[iInc,self.relevantAxes[:]],incInfo[iInc,-1],maxMeshSize,meshSize,sigma])


    ###################################################
    # Method to perform refinement between inclusions #
    ###################################################
    def interInclusionRefinement(self,incInfo):
        """Method to perform refinement between inclusions

        Within this method, the matrix between close inclusions is refined using
        a tanh-function. This method ensures that the space between inclusions
        comprises the user-defined amount of elements. The applied refinement
        function of type "tanh" is described in the function definition of
        "_tanhRefinement()".

        Parameters:
        -----------
        incInfo: array
            extended inclusionInfo array containing information on inclusions
            within the RVE model as well as outside but close to the model
            boundaries
        """
        # get required refinement options
        refinementOptions=self.refinementOptions
        nElemsBetween=refinementOptions["elementsBetweenInclusions"]            # number of elements between inclusion combinations that are considered "close"
        transitionElements=refinementOptions["transitionElements"]              # number of transitioning elements for the continuous jump to go from h_min to h_max
        aspectRatio=refinementOptions["aspectRatio"]                            # aspect ratio of inclusion distance and perpendicular directions
        maxMeshSize=refinementOptions["maxMeshSize"]
        if transitionElements=="auto":                                          # number of transitioning elements is set to "auto"
            transitionElements=nElemsBetween                                    # -> use number of elements between inclusions as default/automatically calculated value
        if refinementOptions["inclusionRefinement"]==True:                      # refinement of inclusions is active and has already been performed
            minMeshSizes=2*np.pi*incInfo[:,[3]]/refinementOptions["elementsPerCircumference"] # -> calculate minimum mesh sizes for the individual inclusions
        else:                                                                   # refinement of inclusions is not active
            minMeshSizes=refinementOptions["maxMeshSize"]*np.ones((np.shape(incInfo)[0],1)) # -> set minimum mesh size for each inclusion to maxMeshSize

        # get relevant axes for distance calculations
        axes=self.relevantAxes

        # loop over all inclusions
        for iInc in range(0,np.shape(incInfo)[0]):

            thisInc=incInfo[iInc,:]                                             # inclusion information for the inclusion under consideration
            otherIncs=incInfo[iInc+1:,:]                                        # inclusion information of other inclusions (prevent double placement of refinement information by only considering inclusion combinations that have not been considered so far)

            # check distance to all remaining other inclusions
            distIncs=self._getDistanceVector(thisInc,otherIncs,axes=axes)       # get center-center distance vectors (per direction) to other inclusions
            normDistIncs=np.linalg.norm(distIncs,axis=1,keepdims=True)          # get norm of center-center distance vectors
            normDistIncBnds=normDistIncs-otherIncs[:,[3]]-thisInc[3]            # get norm of boundary-boundary distance vectors

            # decide which inclusion combinations have to be refined
            # -> Here, the maximum of the minimum mesh densities of the current
            # -> inclusion combinations is used to check whether - with this
            # -> mesh density (plus safety coefficient) - the required amount
            # -> of elements between the inclusions can be ensured
            incsForRefinement=np.array(np.where( (normDistIncBnds.flatten()<=1.1*nElemsBetween*np.maximum(minMeshSizes[iInc,[0]],minMeshSizes[iInc+1:,[0]]).flatten()) & (normDistIncBnds.flatten() > 0) ))

            # loop over all inclusion combinations that have to be refined
            for iRefine in incsForRefinement.flatten():
                refineCenter=thisInc[axes]+0.5*distIncs[iRefine,:]              # get center of refinement (half the distance between the inclusions)
                meshSize=normDistIncBnds[iRefine]/nElemsBetween                 # get required mesh size (distance diveded by required amount of elements)
                refineWidth=normDistIncBnds[iRefine]/2+transitionElements/2*meshSize # get refinement width, i.e. offset of tanh-function to jump from minimum to maximum value (allow coarsening within "transitionElems" elements)
                C=self._getTransformationMatrix(distIncs[iRefine,:],axes)      # get transformation matrix to rotated system between inclusions under consideration

                # set refinement field
                self._setMathEvalField("tanh",np.r_[C.reshape(C.size),refineCenter,refineWidth,maxMeshSize,meshSize,5.3/(nElemsBetween*meshSize),aspectRatio])



################################################################################
#           ADDITIONAL PRIVATE/HIDDEN METHODS FOR INTERNAL USE ONLY            #
################################################################################

    ####################################################################
    # Method to automatically calculate a reasonable maximum mesh size #
    ####################################################################
    def _calculateMaxMeshSize(self):
        """Internal method to calculate the maximum mesh size"""
        if self.size is None:                                                   # RVE size has not been set
            return self.getGmshOption("Mesh.CharacteristicLengthMax")           # -> use Gmsh default setting of maximum mesh size
        else:                                                                   # RVE size is set
            return np.amax(self.size)/10                                        # -> ensure at least 10 elements along the longest edge of the RVE


    #######################################################################
    # Method to get distance between a point and an array of other points #
    #######################################################################
    def _getDistanceVector(self,pointToCheck,pointsToCheckWith,axes=np.arange(0,3)):
        """Internal method to get distances (per direction) between points

        Parameters:
        -----------
        pointToCheck: array
            coordinates of the reference point to calculate the distance for

        pointsToCheckWith: array
            coordinates of all points the distance of the reference point should
            be calculated to

        axes: array
            axes to check the distance for
        """
        distances=pointsToCheckWith[:,axes]-pointToCheck[axes]                  # calculate distance vector in relevant axes directions
        return distances                                                        # return distance vector


    ##############################################################################
    # Method to check collision of an inclusion and an array of other inclusions #
    ##############################################################################
    def _checkIncDistance(self,thisIncInfo,incInfo,axes=np.arange(0,3)):
        """Internal method to check the distance between inclusions.

        Parameters:
        -----------
        thisIncInfo: array
            array containing information (center and radius) of the reference
            inclusion

        incInfo: array
            array containing information (center and radius) of all inclusions
            the distance should be checked for

        axes: array
            axes to check the distance for
        """
        distCenters=np.linalg.norm(self._getDistanceVector(thisIncInfo,incInfo,axes),axis=1)               # get distance of center points (norm)
        if np.any(distCenters-incInfo[:,3]-thisIncInfo[3] <= np.amax(np.r_[incInfo[:,5],thisIncInfo[5]])):  # distance (boundary-boundary) to other inclusions is lower than allowed minumum
            return False                                                        # -> do not accept current inclusion
        else:                                                                   # distance is not lower than allowed minimum
            return True                                                         # -> accept the current inclusion


    #####################################################################
    # Method to check collision of an inclusion with the RVE boundaries #
    #####################################################################
    def _checkBndDistance(self,thisIncInfo,axes=np.arange(0,3)):
        """Internal method to check the distance of inclusions to the surrounding domain.

        In this method, the distance of the current inclusion to the domain boundaries
        is checked. For box-shaped domains, this distance calculation can be performed
        by checking the inclusion distance (per direction) to the two points defining the
        bounding box of the domain.

        Parameters:
        -----------
        thisIncInfo: array
            array containing information (center and radius) of the inclusion

        axes: array
            axes to check the distance for
        """

        # define points to check distance to (domain bounding-box points)
        bndPoints=np.asarray([[0, 0, 0], self.size])                            # temporariliy assume an RVE with origin at [0,0,0] to simplify calculations

        # get distance of center point to boundary points (absolute value for all relevant directions)
        distBndCenter=np.absolute(self._getDistanceVector(thisIncInfo,bndPoints,axes))

        # calculate distance of inclusion boundary to domain boundaries (signed value for all directions)
        distBnd=distBndCenter-thisIncInfo[3]                                    # distance to planes defining the RVE boundary
        distCorner=np.linalg.norm(np.amin(distBndCenter,axis=0))-thisIncInfo[3] # distance to closest corner of the RVE

        # check if distance of inclusion to boundaries is too small
        if np.any(np.absolute(distBnd) <= thisIncInfo[4]) or (np.absolute(distCorner) <= thisIncInfo[4]):
            acceptInc=False
        else:
            acceptInc=True

        # return relevant data
        return acceptInc, distBnd


    ############################################################################
    # Method to get transformation matrix into local system between inclusions #
    ############################################################################
    def _getTransformationMatrix(self,d,axes):
        """Internal method to calculate transformation from global x-y-z coordinate
        system to rotated local system between two inclusions

        Parameters:
        -----------
        d: array
            distance vector between the two inclusions

        axes: array
            axes to check the distance for
        """

        # distinguish 2- and 3-dimensional problems
        if len(axes)==2:                                                        # 2D cylindrical transformation -> rotate system in relevant plane without touching 3rd axis
            phi=np.arctan2(d[1],d[0])                                           # get angle in relevant plane
            C=np.array([[np.cos(phi), np.sin(phi)],[-np.sin(phi), np.cos(phi)]])# get transformation matrix according to cylindrical coordinate transformation

        elif len(axes)==3:                                                      # 3D spherical transformation -> rotate in 2 direction and place rotate x-axis in inclusion distance direction
            phi = np.arctan2(d[1], d[0])                                        # get angle of projected distance vector in original x-y-plane
            theta = np.arctan2(np.sqrt(d[0]**2 + d[1]**2), d[2])                # get 2nd inclination angle towards original x-y-plane
            C=np.array([[np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], # get transformation matrix according to spherical coordinate transformation
                        [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)],
                        [-np.sin(phi), np.cos(phi), 0]])

        # return transformation matrix
        return C


    ############################################
    # Method to set MathEval refinement fields #
    ############################################
    def _setMathEvalField(self,type,data):
        """Internal method to set MathEval refinement fields

        Within this method, MathEval refinement fields can be set according to
        the specified type. An extension of the method is possible by simply
        defining new subfunctions which handle the refinement.

        Parameters:
        -----------
        type: string
            type of refinement function that has to be applied

        data: array
            data to pass as parameters to the required refinement function
        """
        # get refinement function string depending on type of refinement
        if type=="gaussian":                                                    # "Gaussian" refinement function
            refineFunction=self._gaussianRefinement(data)                       # -> call corresponding method
        elif type=="tanh":                                                      # "Tanh" refinement function
            refineFunction=self._tanhRefinement(data)                           # -> call corresponding method
        elif type=="const":                                                     # "Const" refinement function
            refineFunction="{0}".format(data)                                   # -> set constant field

        # update list of refinement fields
        self.refinementFields.append({"fieldType": "MathEval", "fieldInfos": { "F": refineFunction}})


    ###########################################################################
    # Method to calculate "Gaussian" MathEval fields for inclusion refinement #
    ###########################################################################
    def _gaussianRefinement(self,data):
        """Internal method to set "Gaussian" refinement fields

        This method defines refinement fields of the following type:

            Spheres:
            h(x1,x2,x3)=h_max-(h_max-h_min)*exp( -1/2* (( sqrt( (x1-x1_0)^2 + (x2-x2_0)^2 + (x3-x3_0)^2 ) -r0)/(b/4))^2 )

            Cylinders/Disks with relevant axes in the local x1-x2-system:
            h(x1,x2)=h_max-(h_max-h_min)*exp( -1/2* (( sqrt( (x1-x1_0)^2 + (x2-x2_0)^2 ) -r0)/(b/4))^2 )

        It represents a refinement which decreases the mesh size from h_max to
        h_min if the distance r from the inclusion center (x1_0,x2_0,x3_0) is
        close to the value r0. The course of the refinement function resembles a
        normal distribution density function with mean value r0 and standard
        deviation sigma. For convenience, the refinement width (relative to the
        inclusion radius) b is used: since the interval +/-2*sigma covers about
        95% of the values in a normal distribution density function, sigma is
        calculated by b/4.


        Parameters:
        -----------
        data: array
            array containing all parameters for the refinement
            -> data=[x1_0, x2_0, (x3_0), r0, h_max, h_min, b]
        """
        axesString=["x", "y", "z"]                                              # define axes string (needed for problems with only 2 relevant axes)
        if len(self.relevantAxes)==2:                                           # problems with 2 relevant axes (Cylinders/Disks)
            refineFunction="{5}-({5}-({6}))*Exp(-1/2*(((Sqrt(({0}-({2}))^2+({1}-({3}))^2)-({4}))/({7}))^2))".format(*[axesString[ax] for ax in self.relevantAxes[:]],*data)
        elif len(self.relevantAxes)==3:                                         # problems with 3 relevant axes (Spheres)
            refineFunction="{4}-({4}-({5}))*Exp(-1/2*(((Sqrt((x-({0}))^2+(y-({1}))^2+(z-({2}))^2)-({3}))/({6}))^2))".format(*data)

        return refineFunction


    #############################################################################
    # Method to calculate "Tanh" MathEval fields for inter-inclusion refinement #
    #############################################################################
    def _tanhRefinement(self,data):
        """Internal method to set "Tanh" refinement fields

        This method defines refinement functions of the following type:

            Spheres:
            h(x1,x2,x3)=(h_max+h_min)/2 + (h_max-h_min)/2* tanh( m*( ( sqrt( ( C_1k*(xk-x_k0) )^2 ) + aspect^2*( C_2k*(xk-x_k0) )^2 + epsilon^2*( C_3k*(xk-x_k0) )^2 ) -r0) )

            Cylinders/Disks:
            h(x1,x2)=(h_max+h_min)/2 + (h_max-h_min)/2* tanh( m*( ( sqrt( ( C_1k*(xk-x_k0) )^2 ) + aspect^2*( C_2k*(xk-x_k0) )^2 ) -r0) )

        It represents a refinement within the matrix between close inclusions
        which tries to ensure the requested amount of elements and performs a
        "continuous" jump from h_min to h_max, when the distance from the origin
        (x1_0,x2_0,x3_0) reaches the value r0. To allow for different refinement
        widths in the inclusion distance and perpendicular directions, the local
        x1'-x2'-x3'-system is formulated in terms of the global x1-x2-x3 system
        by means of the transformation matrix C_kl. The local x1-axis always points
        in the inclusion distance direction - the downrating of the perpendicular
        axis is performed using the variable aspect. Finally, the width of the
        "jump" is controlled via the initial slope m.

        Parameters:
        -----------
        data: array
            array containing all parameters for the refinement
            -> data=[C_11, C_12, (C_13), C_21, C_22, (C_23), (C_31), (C_32), (C_33), x1_0, x2_0, (x3_0), r0, h_max, h_min, m, aspect]
        """
        if len(self.relevantAxes)==2:
            axesString=["x", "y", "z"]
            refineFunction="({9}+({10}))/2+({9}-({10}))/2*Tanh((Sqrt((({2})*({0}-({6}))+({3})*({1}-({7})))^2+({12})^2*(({4})*({0}-({6}))+({5})*({1}-({7})))^2)-({8}))*({11}))".format(*[axesString[ax] for ax in self.relevantAxes[:]],*data)
        elif len(self.relevantAxes)==3:
            refineFunction="({13}+({14}))/2+({13}-({14}))/2*Tanh((Sqrt((({0})*(x-({9}))+({1})*(y-({10}))+({2})*(z-({11})))^2+({16})^2*(({3})*(x-({9}))+({4})*(y-({10}))+({5})*(z-({11})))^2+({16})^2*(({6})*(x-({9}))+({7})*(y-({10}))+({8})*(z-({11})))^2)-({12}))*({15}))".format(*data)

        return refineFunction
