################################################################################
#    CLASS DEFINITION FOR MESHING MODELS GENERATED USING THE GMSH-PYTHON-API   #
################################################################################
# Within this file, the generic Model class is defined. It is the base class
# for other, more specific classes which aim to mesh models using the Gmsh-
# Python-API. In addition to the methods defined within the Gmsh-Python-API,
# this class provides methods for all basic steps of a model generation using
# Gmsh: some of these methods are only placeholders here and - if required -
# have to be specified/overwritten for the more specialized models.

###########################
# Load required libraries #
###########################
# Standard Python libraries
import os                                                                       # os for file handling (split extensions from file)
import inspect                                                                  # inspect to search for classes in modules
import datetime as dt                                                           # datetime for time stamps
import copy as cp                                                               # copy for deepcopies of arrays
import tempfile as tf                                                           # tempfile for generation of temprory files and folders
import numpy as np                                                              # numpy for array computations
import pickle                                                                   # pickle for saving and loading of gmshModels
import logging                                                                  # logging for log messages
logger=logging.getLogger(__name__)                                              # set logger

# additional program libraries
import gmsh                                                                     # Gmsh Python-API
import meshio                                                                   # meshio for mesh file format conversions

# self-defined class definitions and modules
from ..Geometry import GeometricObjects as geomObj                              # classes for implemented geometric objects
from ..Visualization.GeometryVisualization import GeometryVisualization, PYTHONOCC_AVAILABLE   # class for geometry visualization
from ..Visualization.MeshVisualization import MeshVisualization                 # class for mesh visualization
from ..MeshExport import FeapExport


#############################
# Set configuration options #
#############################
SUPPORTED_GEOMETRY_FORMATS=[".brep", ".stp", ".step"]                           # set supported geometry formats
SUPPORTED_MESH_FORMATS=list(meshio.extension_to_filetype.keys())                # set supported mesh file formats


#############################
# Define GenericModel class #
#############################
class GenericModel:
    """Generic class for meshing models generated using the Gmsh-Python-API

    This class provides the basic mesh generation framework for Gmsh. It
    implements the methods for:

        (1) Setting up a geometry using basic geometric entities and boolean
            operations
        (2) Adding the geometric objects to Gmsh, performing the boolean operations,
            defining physical groups
        (3) creating a mesh with user-defined refinement fields
        (4) saving and visualizing the mesh

    Some of the methods used within the framework are only defined as placeholder
    methods here and have to be specified in detail within the child classes.

    Attributes:
    -----------
    dimension: int
        dimension of the model instance

    modelName: string
        name of the Gmsh model and default name for all resulting files

    gmshConfigChanges: dict
        dictionary for user updates of the default Gmsh configuration

    geometricObjects: list
        list containing the instances of geometric objects used for the
        model geometry creation

    groups: dict
        dictionary with group information for the model entities

    booleanOperations: list
        list with dictionaries defining the individual boolean operations
        to perform for the model generation

    physicalGroups: list
        list with dictionary defining which Gmsh entities are defined as
        physical groups (e.g. different materials)

    refinementFields: list
        list of dictionaries defining the refinement fields that have to
        be added to the Gmsh model

    backgroundField: int
        number of the field that has to be used as the background field/mesh
        for the mesh generation
    """

    #########################
    # Initialization method #
    #########################
    def __init__(self,dimension=None,gmshConfigChanges={}):
        """Initialization method of a generic GmshModel object

        Parameters:
        -----------
        dimension: int
            dimension of the model instance

        gmshConfigChanges: dict
            dictionary for user updates of the default Gmsh configuration
        """
        # set unique model name
        self.modelName="Model_"+dt.datetime.now().strftime("%Y%m%d_%H%M%S%f")   # use time stamp of initialization (with microseconds) for unique model name

        # set default file extensions depending on the savin method
        self._defaultFileExts={
            "Geometry": ".brep",                                                # use ".brep" as default extension for saving the geometry
            "Mesh": ".msh",                                                     # use ".msh" as default extension for saving meshes
            "Model": ".gmshModel",                                              # use ".gmshModel" as default extension for saving models
            "Misc": ".txt"                                                      # use ".txt" as default extension for miscellaneous information
        }

        # initialize Gmsh-Python-API
        self.gmshConfigChanges={                                                # default Gmsh configuration changes
            "General.Terminal": 0,                                              # deactivate console output by default (only activated for mesh generation)
            "Geometry.OCCBoundsUseStl": 1,                                      # use more accurate computation of bounding boxes (slower but advantegous for periodicity constraints)
            "Geometry.Tolerance": 1e-12                                         # adjust geometric tolerance to be a little more precise then default (1e-8)
        }
        self.gmshAPI=self.initializeGmsh(gmshConfigChanges)                     # this assignment facilitates the usage of all methods provided by the gmsh.model class

        # initialize attributes that all instances of GenericModel should have
        self.dimension=dimension                                                # set (highest) dimension of the model
        self.geometricObjects=[]                                                # initialze empty list of geomtric objects (used for model generation)
        self.groups={}                                                          # initialize empty dictionary of groups (used for boolean operations adn physical groups)
        self.booleanOperations=[]                                               # initialize empty list of defined boolean operations (used to generate the model from basic geometrical objects)
        self.physicalGroups=[]                                                  # initialize empty list of defined physical groups (used to identify materials and boundaries within the mesh)
        self.refinementFields=[]                                                # initialize empty list of refinement fields (used to control the mesh sizes)
        self.backgroundField=None                                               # initialize background field for the meshing algorithm (used to control the mesh sizes)


################################################################################
#                MAIN METHODS FOR MODEL AND MESH GENERATION                    #
################################################################################

    ############################################
    # Method to initialize the Gmsh-Python-API #
    ############################################
    def initializeGmsh(self,gmshConfigChanges={}):
        """Gmsh initialization method

        This method initializes the Gmsh-Python-API and adds it to the GmshModel

        Parameters:
        -----------
        gmshConfigChanges: dict
            dictionary with Gmsh configuration options that have to be set
        """
        gmsh.initialize('',False)                                               # initialize Gmsh Python-API without using local .gmshrc configuration file
        self.updateGmshConfiguration(gmshConfigChanges)                         # update default configuration with user updates and set the options
        gmshAPI=gmsh.model                                                      # define gmshAPI as the model class of the Gmsh-Python-API (contains only static methods -> no instance required)

        gmshAPI.add(self.modelName)                                             # add new model to the gmshAPI
        return gmshAPI                                                          # retrun gmshAPI


    ################################################################
    # Method to set up model information and create the Gmsh model #
    ################################################################
    def createGmshModel(self,**geometryOptions):
        """Method to create the Gmsh Model and provide necessary information to it

        This method contains the basic Gmsh model creation steps: after geoetric
        objects are defined, boolean operations are performed to generate the
        final geometry. Parts of the geometry are combined to physical groups in
        order to be able to assign, e.g., material parameters to them. If required,
        a periodicity constraint is finally added to the model.

        Parameters:
        -----------
        geometryOptions: key-value pairs of options
            key-value pairs of options required for the geometry generation
            process
        """

        # define geometric objects and add them to the Gmsh model
        self.defineGeometricObjects(**geometryOptions)                          # placeholder method: has to be specified/overwritten for the individual models
        self.addGeometricObjectsToGmshModel()                                   # use Gmsh-API to add geometric information to the Gmsh model

        # define boolean operations and add them to the Gmsh model (perform them)
        self.defineBooleanOperations()                                          # placeholder method: has to be specified/overwritten for the individual models
        self.performBooleanOperationsForGmshModel()                             # use Gmsh-API to perform defined operations

        # define physical groups and add them to the Gmsh model
        self.definePhysicalGroups()                                             # placeholder method: has to be specified/overwritten for the individual models
        self.addPhysicalGroupsToGmshModel()                                     # use Gmsh-API to add defined groups to the Gmsh model

        # set up periodicity constraints
        self.setupPeriodicity()                                                 # placeholder method: has to be specified/overwritten for the individual models if necessary


    ####################################################################
    # Method to calculate refinement information and generate the mesh #
    ####################################################################
    def createMesh(self,threads=None,refinementOptions={}):
        """Method to generate the model mesh

        This method contains the basic mesh generation steps for a Gmsh model:
        refinement fields are calculated with user-defined options and added to
        the model. Afterwards, a background field is specified and used for the
        mesh size computation within Gmsh. Finally, the mesh is created.

        Parameters:
        -----------
        threads: int
            number of threads to use for the mesh generation
        refinementOptions: dict
            dictionary with user-defined options for the refinement field calculations
        """

        if threads is not None:                                                 # set number of threads in Gmsh
            self.updateGmshConfiguration({"Mesh.MaxNumThreads1D": threads,
                                          "Mesh.MaxNumThreads2D": threads,
                                          "Mesh.MaxNumThreads3D": threads})

        # deine refinement information and add them to the Gmsh model
        self.defineRefinementFields(refinementOptions=refinementOptions)        # placeholder method: has to be specified/overwritten for the individual models
        self.addRefinementFieldsToGmshModel()                                   # use Gmsh-API to add defined fields to the Gmsh model

        # set background field for meshing procedure (if possible)
        if not self.backgroundField is None:
            self.gmshAPI.mesh.field.setAsBackgroundMesh(self.backgroundField)   # define background field (field which is used for mesh size determination)

        # generate mesh (with activate console output)
        self._gmshOutput(1)                                                     # activate Gmsh console output
        self.gmshAPI.mesh.generate(self.dimension)                              # generate mesh using the Gmsh-API
        self._gmshOutput(0)                                                     # deactivate Gmsh console output


    ############################################################
    # Closing method to terminate the current Gmsh-API session #
    ############################################################
    def close(self):
        """Gmsh finalization method

        The Gmsh-Python-API has to be finalized for a proper termination of the
        model.
        """
        gmsh.finalize()



################################################################################
#             MAIN METHODS FOR LOADING AND SAVING INFORMATION                  #
################################################################################

    ########################################################
    # Method to save the geometry of the model into a file #
    ########################################################
    def saveGeometry(self,file=None):
        """Method to save the generated geometry into a geometry file

        This method allows to store geometry information into ".step" or ".brep"-
        files.
        """
        # get fileparts of passed file string (return defaults if nothing is passed)
        fileDir,fileName,fileExt=self._getFileParts(file,"Geometry")

        if fileExt in SUPPORTED_GEOMETRY_FORMATS:                               # check if file extension is supported by gmsh
            gmsh.write(fileDir+"/"+fileName+fileExt)                            # write geometry to file (use Gmsh-internal "guess from extension" feature)
        else:
            raise ValueError("Unknown geometry file extension {}. The output geometry format must be supported by the gmsh library.".format(fileExt))


    ####################################################
    # Method to save the mesh of the model into a file #
    ####################################################
    def saveMesh(self,file=None):
        """Method to save the generated mesh into a mesh file

        After the mesh is generated, it has to be saved into a usable file format.
        Here, all meshes that are supported by the meshio library, can be used to
        save the mesh. If meshio is not available, the mesh format is restricted

        """
        # get fileparts of passed file string (return defaults if nothing is passed)
        fileDir,fileName,fileExt=self._getFileParts(file,"Mesh")

        # create mesh file depending on the chosen file extension
        os.makedirs(fileDir,exist_ok=True)                                      # ensure that the file directory exists
        if fileExt == ".msh":                                                   # file extension is ".msh"
            gmsh.write(fileDir+"/"+fileName+fileExt)                            # -> save mesh using built-in gmsh.write method
        elif fileExt == '.msh2':                                                # file extension is ".msh2"
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.)                    # change format to msh2
            gmsh.write(fileDir+"/"+fileName+".msh")                             # -> save mesh using built-in gmsh.write method
        elif fileExt == ".feap":                                                # file extension is ".feap" -> write feap mesh files
            FeapExport(self)
        else:                                                                   # file extension is different from ".msh"
            if fileExt in SUPPORTED_MESH_FORMATS:                               # -> check if file extension is supported by meshio
                with tf.TemporaryDirectory() as tmpDir:                         # ->-> create temporary directory
                    tmpFile=tmpDir+"/"+self.modelName+".msh"                    # ->-> create temporary file
                    gmshBinaryConfig=self.getGmshOption("Mesh.Binary")          # ->-> get Gmsh configuration for binary mesh export
                    self.setGmshOption("Mesh.Binary",1)                         # ->-> temporarily activate binary mesh export (reduce file size, increase speed)
                    gmsh.write(tmpFile)                                         # ->-> use built-in gmsh.write method to generate binary mesh in temporary folder
                    self.setGmshOption("Mesh.Binary",gmshBinaryConfig)          # ->-> reset Gmsh configuration
                    self._convertMesh(tmpFile,fileDir+"/"+fileName+fileExt)     # ->-> convert mesh to required file format
            else:                                                               # raise error if mesh file format is not supported by meshio
                raise ValueError("Unknown mesh file extension {}. The output mesh format must be supported by the meshio library.".format(fileExt))


    #####################################################
    # Method to save Gmsh model object to a binary file #
    #####################################################
    def saveModel(self,file=None):
        """Method to save the complete model into a pickle object

        In order to be able to reuse generated models, the whole model can be
        saved. Within this method, the pickle module is used to save the model
        to a binary file.
        """
        # get file parts of passed file string (return defaultsd if nothing is passed)
        fileDir,fileName,fileExt=self.__getFileParts(file,"Model")

        # save file
        os.makedirs(fileDir,exist_ok=True)                                      # ensure that the file directory exists
        with open(fileDir+"/"+fileName+fileExt,"wb") as file:                   # open file with writing permissions in binary mode
            pickle.dump(self,file)                                              # save file using pickle


    #################################################
    # Class method to load existing model instances #
    #################################################
    @classmethod                                                                # define method as a class method
    def load(cls,fileName):
        """Method to load an existing GmshModel object

        Objects that have been saved to a binary file using the pickle module
        can be reloaded here.
        """
        with open("fileName","rb") as file:                                     # load file
            return pickle.load(file)                                            # load saved file with pickle module



################################################################################
#                           VISUALIZATION METHODS                              #
################################################################################

    ##########################################################
    # Method to visualize the model geometry using pythonocc #
    ##########################################################
    def visualizeGeometry(self):
        """Method to visualize the Gmsh model geometry using pythonocc"""
        if PYTHONOCC_AVAILABLE:                                                 # optional pythonocc package is available
            GeometryVisualization(self)                                         # -> visualize the geometry
        else:                                                                   # optional pythonocc package is unavailable
            logger.warning("Geometry visualization is unavailabe due to missing packages.") # do nothing but printing a warning


    ############################################################
    # Method to visualize the model mesh using pyvista and vtk #
    ############################################################
    def visualizeMesh(self):
        """Method to visualize the generated mesh using pyvista and vtk"""
        MeshVisualization(self)                                                 # -> visualize the mesh


    ####################################################################
    # Method to show the whole model in Gmsh using the Gmsh-Python-API #
    ####################################################################
    def showModelInGmsh(self):
        """Method to open the complete model in Gmsh"""
        gmsh.fltk.run()



################################################################################
#       PLACEHOLDER METHODS TO BE SPECIALIZED FOR THE INDIVIDUAL MODELS        #
################################################################################

    ##################################################
    # Method for the definition of geometric objects #
    ##################################################
    def defineGeometricObjects(self,**options):
        """Placeholder method for the definition of geometric objects. Has to be
        specified in child classes"""
        pass


    ###################################################
    # Method for the definition of boolean operations #
    ###################################################
    def defineBooleanOperations(self):
        """Placeholder method for the definition of necessary boolean Operations.
        Has to be specified in child classes"""
        pass


    ################################################
    # Method for the definition of physical groups #
    ################################################
    def definePhysicalGroups(self):
        """Placeholder method to define required physical groups. Has to be
        specified in child classes"""
        pass


    ###########################################
    # Method to define refinement information #
    ###########################################
    def defineRefinementFields(self):
        """Placeholder method to define/compute refinement fields for the mesh
        generation. Has to be specified in child classes"""
        pass


    ###############################################################
    # Method to set up periodicity constraints for the Gmsh model #
    ###############################################################
    def setupPeriodicity(self):
        """Placeholder method to set up periodicity constraints for RVEs. Has to
        be specified in child classes, if required"""
        pass



################################################################################
#         INTERFACING METHODS TO PASS INFORMATION TO THE GMSH MODEL            #
################################################################################

    ###############################################
    # Method to transform Gmsh entity Tags to IDs #
    ###############################################
    def getIDsFromTags(self,tags):
        """Interfacing method to get Gmsh entity IDs from given list of Gmsh
        entityTags

        Parameters:
        -----------
        tags: (list of) tuples
            list of Gmsh entity tag tuples
            tags=(entityDimension, entityID)
        """
        _,IDs=map(list,zip(*tags))                                              # get IDs of tags-array (tag: (dimension,ID)-tuple)
        return IDs                                                              # return IDs


    ############################################################
    # method to get a single Gmsh option depending on its type #
    ############################################################
    def getGmshOption(self,optionName):
        """Method to get the value of a Gmsh configuratio option with known name

        Parameters:
        -----------
        optionName: string
            name of the option
        """
        try:                                                                    # try to return option value assuming it is a string
            return gmsh.option.getString(optionName)                            # -> use built-in gmsh.option.getString method
        except:                                                                 # option value was no string, so it must be a number
            return gmsh.option.getNumber(optionName)                            # -> use built-in gmsh.option.getNumber method


    ############################################################
    # method to set a single Gmsh option depending on its type #
    ############################################################
    def setGmshOption(self,optionName=None,optionValue=None):
        """Method to set a Gmsh configuration option

        Parameters:
        -----------
        optionName: string
            name of the option to set
        optionValue: int/float/string
            value of the option to set
        """
        if isinstance(optionValue,str):                                         # option value is a string
            gmsh.option.setString(optionName,optionValue)                       # -> use built-in gmsh.option.setString method
        elif isinstance(optionValue,int) or isinstance(optionValue,float):      # optionValue is a number
            gmsh.option.setNumber(optionName,optionValue)                       # -> use built-in gmsh.option.setNumber method


    ############################################################
    # Method to add the model boundary to the physical entites #
    ############################################################
    def getBoundaryEntities(self):
        """Method to get the entities on the boundary of the Gmsh model"""
        # get information for a physical entity containing the model boundary
        return self.gmshAPI.getBoundary(self.gmshAPI.getEntities(self.dimension), combined=True, oriented=False, recursive=False)


    ###########################################
    # Method to update the gmsh configuration #
    ###########################################
    def updateGmshConfiguration(self,configurationUpdate):
        """Method to update the Gmsh configuration options with a dictionary
        of updated options

        Parameters:
        -----------
        configurationUpdate: dict
            dictionary of configuration options to be updated
        """
        self.gmshConfigChanges.update(configurationUpdate)                      # update stored Gmsh configuration
        for optionName, optionValue in self.gmshConfigChanges.items():          # loop over all configuration settings
            self.setGmshOption(optionName=optionName,optionValue=optionValue)   # -> activate changed configuration


    #########################################################
    # Method to add all geometric objects to the Gmsh model #
    #########################################################
    def addGeometricObjectsToGmshModel(self):
        """Method to add Gmsh representations of the gmshModels geometric objects"""
        for obj in self.geometricObjects:                                       # loop over all geometric objects of the model
            gmshTag=obj.addToGmshModel(self.gmshAPI)                            # -> add a Gmsh representation of the object to the model and save the correspondig tag
            self.groups[obj.group].append(gmshTag)                              # -> add tag to the group of the geometric object


    ###############################################################
    # Method to perform all boolean operations for the Gmsh model #
    ###############################################################
    def performBooleanOperationsForGmshModel(self):
        """Method to perform defined boolean operations for the Gmsh model"""

        # loop over all boolean operations of the model
        for booleanOP in self.booleanOperations:

            # get details of the boolean operation to be performed
            operation=self._getBooleanOperation(booleanOP["operation"])
            objectTags=self.groups[booleanOP["object"]]
            toolTags=self.groups[booleanOP["tool"]]
            removeObject=booleanOP["removeObject"]
            removeTool=booleanOP["removeTool"]
            resultingGroup=booleanOP["resultingGroup"]

            # perform boolean operation
            outputTags,outputTagsMap=operation(objectTags,toolTags,tag=-1,removeObject=removeObject,removeTool=removeTool)

            # synchronize OCC-CAD representation with model
            self.gmshAPI.occ.synchronize()

            # update groups
            self.groups[resultingGroup]=outputTags


    ###############################################
    # Method to add physical groups to Gmsh model #
    ###############################################
    def addPhysicalGroupsToGmshModel(self):
        """Method to add defined physical groups to the Gmsh model """

        # loop over all physical entities of the model
        for physGrp in self.physicalGroups:

            # get details of the physical entity to add
            grpDim=physGrp["dimension"]                                         # get dimension of the physical group
            grpName=physGrp["group"]                                            # get the group of the physical entity (used as name)
            grpNumber=physGrp["physicalNumber"]                                 # get the number defined for the physical entity (used as material number)
            grpEntIDs=self.getIDsFromTags(self.groups[grpName])                 # find Gmsh representations of all group members and get IDs from their tags

            # set physical groups
            self.gmshAPI.addPhysicalGroup(grpDim,grpEntIDs,grpNumber)           # define the entity group as physical and set correct physical number
            self.gmshAPI.setPhysicalName(grpDim,grpNumber,grpName)              # set corresponding name of the physical group (equal to name of the group that is declared as physical for simplicity)


    #################################################
    # Method to add refinement fields to Gmsh model #
    #################################################
    def addRefinementFieldsToGmshModel(self):
        """Method to add defined refinement fields to the Gmsh model"""

        # loop over all refinement fields defined for the model
        for refineField in self.refinementFields:

            # get details of the refinement field to add
            fieldType=refineField["fieldType"]                                  # get the type of refinement field
            fieldInfos=refineField["fieldInfos"]                                # get information required for this type of refinement field

            # set refinement field
            fieldTag=self.gmshAPI.mesh.field.add(fieldType,tag=-1)              # add new refinement field and save its number
            for optName, optVal in fieldInfos.items():                          # provide all necessary information for this field from fieldInfo dictionary
                if isinstance(optVal,str):                                      # -> current option value is a string
                    self.gmshAPI.mesh.field.setString(fieldTag,optName,optVal)  # ->-> use built-in setString method of gmsh.model.mesh.field
                elif isinstance(optVal,int) or isinstance(optVal,float):        # -> current option value is a number
                    self.gmshAPI.mesh.field.setNumber(fieldTag,optName,optVal)  # ->-> use built-in setNumber method of gmsh.model.mesh.field
                elif isinstance(optVal,list) or isinstance(optVal,np.ndarray):  # -> current option value is a list or numpy array
                    self.gmshAPI.mesh.field.setNumbers(fieldTag,optName,optVal) # ->-> use built-in setNumbers method of gmsh.model.mesh.field



################################################################################
#           INTERFACING METHODS TO ADD GEOMETRIC OBJECTS TO THE MODEL          #
################################################################################

    ##############################################################
    # Method to add a single geometric object to the Gmsh model #
    ##############################################################
    def addGeometricObject(self,objClassString,**objData):
        """Method to add one of the objects that are defined within the class
        geometricObjects and its child classes to the Gmsh model.

        Parameters:
        -----------
        objClass: class
            class the geometric object is defined in
        objData: keyworded object data
            enumeration of keyworded arguments needed for the creation of the
            new geometric object of class objectClass
        """
        objClass=self._getGeometricObjectClass(objClassString)
        objInstance=objClass(**objData)
        objGroup=objInstance.group
        self.geometricObjects.append(objInstance)
        self.groups.update({objGroup: []}) if objGroup not in self.groups else self.groups


################################################################################
#               PRIVATE/HIDDEN METHODS FOR INTERNAL USE ONLY                   #
################################################################################

    ################################################
    # Method to check file string for saving files #
    ################################################
    def _getFileParts(self,fileString,whatToSave):
        """Internal method to get the file directory, name and extension from
        a fileString

        Parameters:
        -----------
        fileString: string
            string to analyze
        whatToSave: string
            flag that indicates whether "geometry", "mesh" or "model" have to
            be saved
        """

        # check if no fileString was passed
        if fileString is None:                                                  # no file string passed
            fileString=""                                                       # -> set empty file string

        # process passed file string
        fileDir,fileName= os.path.split(fileString)                             # get file directory and name (with extension)
        fileName,fileExt=os.path.splitext(fileName)                             # split file name into name and extension
        if fileDir == "":                                                       # check if file directory is empty
            fileDir="."                                                         # -> set default value (current directory)
        if fileName == "":                                                      # check if file name is empty
            fileName=self.modelName                                             # use unique model name as file name
        if fileExt == "":                                                       # check if file extension is empty
            fileExt=self._defaultFileExts[whatToSave]                           # -> set default value (.gmshModel)

        # return file parts
        return fileDir, fileName, fileExt


    ########################################
    # Method to toggle Gmsh console output #
    ########################################
    def _gmshOutput(self,switch):
        """Method to enable/disable Gmsh console output"""
        self.setGmshOption(optionName="General.Terminal",optionValue=switch)


    ############################################################################
    # Method to return the correct boolean operation from the operation string #
    ############################################################################
    def _getBooleanOperation(self,operation):
        """Internal method to return the correct boolean operation function
        from an operation string

        Parameters:
        -----------
        operation: string
            boolean operation to be performed
        """
        if operation == "cut":                                                  # operation to be performed is "cut"
            return self.gmshAPI.occ.cut                                         # -> use built-in gmsh.model.occ.cut method
        elif operation == "fuse":                                               # operation to be performed is "fuse"
            return self.gmshAPI.occ.fuse                                        # -> use built-in gmsh.model.occ.fuse method
        elif operation == "fragment":                                           # operation to be performed is "fragment"
            return self.gmshAPI.occ.fragment                                    # -> use built-in gmsh.model.occ.fragment method
        elif operation == "intersect":                                          # operation to be performed is "intersect"
            return self.gmshAPI.occ.intersect                                   # -> use built-in gmsh.model.occ.intersect method
        else:                                                                   # operation to be performed is something different
            raise ValueError("Unknown boolean operation {}".format(operation))  # -> raise error that type of boolean operation is not known


    #########################################################
    # Method to get defined geometric objects from a string #
    #########################################################
    def _getGeometricObjectClass(self,objString):
        """Internal method to return the correct geometric object class from
        an object class string

        Parameters:
        -----------
        objString: string
            required geometric object class (as a string)
        """
        for objKey, objClass in inspect.getmembers(geomObj,inspect.isclass):    # get all classes from the geometricObjects file
            if objKey == objString:                                             # check if class key matches the object string that was passed
                return objClass                                                 # return class


    ###########################################################
    # Method to calculate the overall Gmsh model bounding box #
    ###########################################################
    def _getGmshModelBoundingBox(self):
        """Internal method to get the overall Gmsh model bounding box"""
        entityBBoxes=np.atleast_2d([self.gmshAPI.getBoundingBox(*entTag) for entTag in self.gmshAPI.getEntities()])
        modelBBox= np.array([[*np.amin(entityBBoxes[:,0:3],axis=0)], [*np.amax(entityBBoxes[:,3:6],axis=0)]])
        return modelBBox


    ##########################################################
    # Method to convert meshes into meshio-supported formats #
    ##########################################################
    def _convertMesh(self,inFile,outFile):
        """Internal method to convert meshes between different file formats
        using meshio

        Parameters:
        -----------
        inFile: string
            file string (directory/name.extension) for the input mesh file
        outFile: string
            file string (directory/name.extension) for the output mesh file
        """
        mesh=meshio.Mesh.read(inFile)                                           # read file of mesh to convert with meshio (get mesh format from file extension)
        mesh.write(outFile)                                                     # write file for converted mesh (get mesh format from file extension)
