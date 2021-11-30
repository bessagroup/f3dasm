################################################################################
#             CLASS FOR RVE VISUALIZATION USING PYVISTA AND VTK                #
################################################################################
# This file provides the class definition for the mesh visualization of a Gmsh
# model. It is based on the pyvista and vtk libraries und implements additional
# features for the mesh visualization.

############################
# Load required libraries: #
############################
# Standard Python libraries
import numpy as np                                                              # numpy for fast array computations
import copy as cp                                                               # copy for deepcopies
import tempfile as tf                                                           # tempfile for temporary files and folders
import logging                                                                  # logging for warning messages
logger=logging.getLogger(__name__)                                              # -> set logger

# additional program libraries
import pyvista as pv                                                            # -> load pyVista visualization module
import vtk                                                                      # -> load vtk since pyvista depends on it, i.e. it must be available


##################################
# Define MeshVisualization class #
##################################
class MeshVisualization():
    """Class for Gmsh model mesh visualization

    This class visualizes the mesh of a given GmshModel using the pyvista and
    vtk libraries. It provides a simple method to visualize the model geometry
    by storing the mesh into a temporary ".vtu" file which is read by pyvista.
    With additional slider widgets for a threshold algorithm based on physical
    groups of the Gmsh model and a box widget to facilitate an extraction of
    areas within the mesh, the visualization tool allows to get an impression
    of the generated mesh.

    Attributes:
    -----------
    model: GmshModel object instance
        model instance for which the geometry has to be displayed

    defaultSettings: dict
        dictionary holding the default widget visualization options to allow
        restoring those options

    currentSettings: dict
        dictionary holding the current widget visualization options

    plotterObj: pyvista plotter object instance
        current plotter/renderer object

    mesh: mesh object instance
        object holding the current mesh information

    thresholdAlgorithm: vtk Threshold Algorithm instance
        object instance with information for the applied vtkThreshold algorithm

    thresholdAlgorithm: vtk Extraction Algorithm instance
        object instance with information for the applied vtkExtractGeometry
        algorithm

    extractionBox: vtkBox object instance
        object instance storing information of the extraction box

    extractionBoxBounds: pyvista PolyData object instance
        object instance storing information about the extraction box boundaries

    active widgets: list
        list of active widgets within the rendering object
    """
    #########################
    # Initialization method #
    #########################
    def __init__(self,model=None):
        """Initilaization method for the MeshVisualization class

        This method initializes a new instance of the MeshVisualization
        class and visualizes the mesh of the passed GmshModel instance.
        Within the method, basic visualization settings are defined, before the
        visualization method is called.

        Parameters:
        model: GmshModel object instance
            model for which the mesh has to be visualized
        """
        # plausibility checks for input arguments
        if model is None:
            raise TypeError("No model instance to visualize passed. For the visualization of a model, that model has to be known. Check your input data.")
        self.model=model

        # set plotting theme
        pv.set_plot_theme("paraview")                                           # use Paraview style for plotting

        # get necessary model information for the default configuration
        physicalIDs=model.getIDsFromTags(model.gmshAPI.getPhysicalGroups(dim=model.dimension))  # get group IDs for physical groups of the highest dimension
        modelBBox=model._getGmshModelBoundingBox()                              # get the overall bounding box of the Gmsh model

        # define settings
        self.defaultSettings={
                "lowerThreshold": min(physicalIDs),                             # set lower threshold value to minimum value of active scalar field
                "upperThreshold": max(physicalIDs),                             # set upper threshold value to minimum value of active scalar field
                "boxBounds": modelBBox.T.reshape(6)                             # define bounds of extraction box to match bounds of the model (factor 1.25 will be applied automatically)
        }
        self.currentSettings=cp.deepcopy(self.defaultSettings)                  # copy default settings as initial version of current settings

        # initialize arrays required for proper mesh visualization
        self.plotterObj=pv.Plotter(title="GmshModel Mesh Visualization")        # initialize plotterObj from pyvista and set title
        self.mesh=None                                                          # initialize mesh
        self.thresholdAlgorithm=None                                            # initialize threshold algorithm
        self.extractionBox=vtk.vtkBox()                                         # initiliaze dummy extraction box to update information of boxWidget
        self.extractionBoxBounds=pv.PolyData()                                  # initialize dummy extraction box bounds to store information of boxWidget bounds
        self.extractionAlgorithm=None                                           # initialize extraction algorithm
        self.activeWidgets=[]                                                   # initialize list of active widgets

        # visualize the model mesh
        self.visualizeMesh()



################################################################################
#                 MAIN METHODS FOR GEOMETRY VISUALIZATION                      #
################################################################################

    ################################
    # Method to visualize the mesh #
    ################################
    def visualizeMesh(self):
        """Main method for the mesh visualization

        Within this method, the mesh data of the model is stored into a temporary
        ".vtu"-file which is read pyvista. After setting up widgets and additional
        key-press-events, the mesh is visualized.
        """

        with tf.TemporaryDirectory() as tmpDir:                                 # create temporary directory
            tmpFile=tmpDir+"/"+self.model.modelName+".vtu"                      # set name of file in temorary directory
            self.model.saveMesh(tmpFile)                                        # create temporary file

            self.mesh=pv.UnstructuredGrid(tmpFile)                              # load mesh from temporary file with pyvista
            self.mesh.set_active_scalars("gmsh:physical")                       # set gmsh:physical to be the active scalar
            self.scalars = self.mesh.active_scalars_info                        # get field ID and name of active scalar field

            # add widgets and key events
            self.addSliderWidgets()                                             # add slider widgets for threshold filter
            self.addBoxWidget()                                                 # add box widget for extraction filter
            self.addKeyPressEvents()                                            # add defined key press events

            # add mesh to plotterObj
            self.plotterObj.add_mesh(self.mesh,scalars=self.scalars[1],name="mesh",reset_camera=False,clim=[self.currentSettings["lowerThreshold"],self.currentSettings["upperThreshold"]],show_edges=True)

            # update settings of the rendering scene and show plot
            self.plotterObj.remove_scalar_bar()                                 # remove scalar bar (since no results will be shown but just the mesh)
            self.plotterObj.show_axes()                                         # show axes orientation
            self.plotterObj.show()                                              # show resulting plot


    #######################################
    # Method to apply filters to the mesh #
    #######################################
    def filterPipeline(self):
        """Filter pipeline method for mesh visualization

        For every confirmed change of the widget settings, the mesh has to be
        updated. To this end, the filter pipeline consisting of the implemented
        threshold and extraction algorithms has to be executed. It is done
        within this method.
        """

        # apply threshold filter
        self.thresholdAlgorithm.ThresholdBetween(self.currentSettings["lowerThreshold"],self.currentSettings["upperThreshold"]) # set threshold values from slider widgets
        self.thresholdAlgorithm.Update()                                        # update information stored in thresold algorithm
        self.mesh.shallow_copy(self.thresholdAlgorithm.GetOutput())             # store resulting information in the mesh -> update mesh to visualize

        # apply extraction filter
        self.extractionBox.SetBounds(*self.currentSettings["boxBounds"])        # set box bounds from box widget
        self.extractionAlgorithm.Update()                                       # update information stored in extraction algorithm
        self.mesh.shallow_copy(self.extractionAlgorithm.GetOutput())            # store resulting information in the mesh -> update mesh to visualize

        # force update of the rendering scene
        self.plotterObj.render()                                                # if not done, the renderer might update only after the next interaction with the plot



################################################################################
#                      INTERFACING METHODS FOR PYVISTA                         #
################################################################################

    ############################################
    # Method to set up the threshold algorithm #
    ############################################
    def setupThresholdAlgorithm(self):
        """Threshold algorithm method for mesh visualization

        This method implements the threshold algorithm used within the filter
        pipeline. Based on the settings of the slider widgets, the visibility
        of physical groups is enabled oder disabled.
        """
        self.thresholdAlgorithm = vtk.vtkThreshold()                            # use vtkThreshold filter class of vtk library
        self.thresholdAlgorithm.SetInputDataObject(self.mesh)                   # set mesh to plot as input data object
        self.thresholdAlgorithm.SetInputArrayToProcess(0, 0, 0, self.scalars[0].value, self.scalars[1]) # args: (idx, port, connection, field, name)
        self.thresholdAlgorithm.Update()                                        # update threshold algorithm
        self.mesh=pv.wrap(self.thresholdAlgorithm.GetOutput())                  # update mesh once and connect it with with the algorithm output


    #############################################
    # Method to set up the extraction algorithm #
    #############################################
    def setupExtractionAlgorithm(self):
        """Extraction algorithm method for mesh visualization

        This method implements the extraction algorithm used within the filter
        pipeline. Based on the settings of the box widget, the mesh elements
        within the extraction box bounds are extracted.
        """
        self.extractionAlgorithm = vtk.vtkExtractGeometry()                     # use vtkExtractGeometry filter class of vtk library
        self.extractionAlgorithm.SetInputDataObject(self.mesh)                  # use (potentially updated) mesh to plot as input data object
        self.extractionAlgorithm.SetImplicitFunction(self.extractionBox)        # describe extraction box by implicit function with the information stored in extractionBox
        self.extractionAlgorithm.Update()                                       # update extraction algorithm


    ################################
    # Method to add slider widgets #
    ################################
    def addSliderWidgets(self):
        """Method to configure a slider widget and add it to the rendering scene"""

        # set up corresponding algorithm
        self.setupThresholdAlgorithm()                                          # connect threshold algorithm to slider widgets

        # set up widgets
        pointsA=np.array([[0.01, 0.92],[0.01, 0.79]])                           # define starting points of slider widgets (normalized coordinates)
        pointsB=np.array([[0.11, 0.92],[0.11, 0.79]])                           # define end points of slider widgets (normalized coordinates)
        initialValues=[self.currentSettings["lowerThreshold"],self.currentSettings["upperThreshold"]] # define initial slider values according to current settings
        titles=["Min", "Max"]                                                   # define titles of slider widgets
        callbacks=[self._callbackLowerThreshold, self._callbackUpperThreshold]  # define callback methods of the slider widgets
        for iSlider in [0,1]:                                                   # loop over both sliders
            slider=self.plotterObj.add_slider_widget(callback=callbacks[iSlider], value=initialValues[iSlider], rng=initialValues, title=titles[iSlider], color=[1,1,1]) # use add_slider_widget of pyvista to define the widgets
            slider.GetRepresentation().GetPoint1Coordinate().SetValue(pointsA[iSlider,0], pointsA[iSlider,1]) # properly set slider widget starting point (error in pyvista implementation (divide by int))
            slider.GetRepresentation().GetPoint2Coordinate().SetValue(pointsB[iSlider,0], pointsB[iSlider,1]) # properly set slider widget end point (error in pyvista implementation (divide by int))
            slider.GetRepresentation().SetSliderWidth(0.02)                     # set width of the slider (normalized coordinates)
            slider.GetRepresentation().SetSliderLength(0.015)                   # set length of the slider (normalized coordinates)
            slider.GetRepresentation().SetEndCapLength(0.005)                   # set length of the bars at beginning and end of the slider widget (normalized coordinates)
            slider.GetRepresentation().SetEndCapWidth(0.033)                    # set width of the bars at beginning and end of the slider widget (normalized coordinates)
            slider.GetRepresentation().SetTubeWidth(0.005)                      # set width of the line connecting both slider ends (normalized coordinates)
            slider.GetRepresentation().SetTitleHeight(0.025)                    # set height of the title (normalized coordinates)
            slider.GetRepresentation().SetLabelHeight(0.025)                    # set height of the label (normalized coordinates)
            slider.GetRepresentation().GetSelectedProperty().SetColor([0.9, 0.9, 0.95]) # set color of selected slider
            slider.Off()                                                        # initially make slider widget invisible
            self.activeWidgets.append(slider)                                   # add slider widget to the list of active widgets


    ############################
    # Method to add box widget #
    ############################
    def addBoxWidget(self):
        """Method to configure a box widget and add it to the rendering scene"""

        # set up corresponding algorithm
        self.setupExtractionAlgorithm()                                         # connect extraction algorithm to box widget

        # set up widget
        box=self.plotterObj.add_box_widget(callback=self._callbackExtraction, bounds=self.defaultSettings["boxBounds"], rotation_enabled=False) # use add_box_widget of pyvista to define the widget
        box.Off()                                                               # initially make box widget invisible
        self.activeWidgets.append(box)                                          # add box widget to the list of active widgets


    ##################################
    # Method to add key press events #
    ##################################
    def addKeyPressEvents(self):
        """Method to add all user-defined key-press events"""
        activeKeys=["h", "m", "d", "space", "x", "y", "z"]                      # define list of keys with active (user-defined) key press events
        for key in activeKeys:                                                  # loop over all active keys
            self.plotterObj.add_key_event(key,self._keyPressEvents)             # add corresponding key press events


    ####################################
    # Method to toggle menu visibility #
    ####################################
    def toggleMenu(self):
        """Method to toggle the menu/widget visibility"""
        # loop over all active widgets
        for widget in self.activeWidgets:                                       # get current widget
            toggleFuncs=[widget.On, widget.Off]                                 # define toggle functions depending on state [0/1]
            toggleStatus=widget.GetEnabled()                                    # get widget state
            toggleFuncs[toggleStatus]()                                         # use toggle function according to current state


    ######################################
    # Method to restore default settings #
    ######################################
    def restoreDefaultSettings(self):
        """Method to restore default visualization options"""
        self.currentSettings.update(self.defaultSettings)                       # restore default settings
        self.plotterObj.slider_widgets[0].GetRepresentation().SetValue(self.currentSettings["lowerThreshold"]) # set value of lower threshold slider according to settings
        self.plotterObj.slider_widgets[1].GetRepresentation().SetValue(self.currentSettings["upperThreshold"]) # set value of upper threshold slider according to settings
        self.plotterObj.box_widgets[0].PlaceWidget(*self.currentSettings["boxBounds"])  # set bounds of extraction box according to settings
        self._callbackExtraction(self.extractionBoxBounds)                      # also call callback method of extraction box to complete the update
        self.filterPipeline()                                                   # update filter pipeline with restored settings


    ####################################
    # Method to show command line help #
    ####################################
    def showCommandLineHelp(self):
        """Internal method to show command line help after the rendering window
        is displayed"""
        infoText=("\nUse one of the following key events to control the plot:\n"
                  "\n"
                  "\ts \tactivate surface representation of objects\n"
                  "\tw \tactivate wireframe representation if objects\n"
                  "\tv \tenable isometric view\n"
                  "\tx \tset view to y-z-plane\n"
                  "\ty \tset view to z-x-plane\n"
                  "\tz \tset view to x-y-plane\n"
                  "\tm \ttoggle menu\n"
                  "\tspace \tconfirm menu settings\n"
                  "\td \trestore default settings\n"
                  "\tq \tclose rendering window\n")
        print(infoText)



################################################################################
#           ADDITIONAL PRIVATE/HIDDEN METHODS FOR INTERNAL USE ONLY            #
################################################################################

    #########################################################
    # Internal callback method tofor lower threshold slider #
    #########################################################
    def _callbackLowerThreshold(self,sliderThresholdValue):
        """Internal callback method for the lower threshold widget"""
        self.currentSettings["lowerThreshold"]=sliderThresholdValue             # update lower threshold value with current slider position

        # prevent mixed up threshold sliders (lower > upper)
        try:                                                                    # use try-catch environment to prevent error in plot initialization (other slider is not defined yet)
            if sliderThresholdValue > self.plotterObj.slider_widgets[1].GetRepresentation().GetValue(): # check if value of lower threshold exceeds current upper threshold value
                self.plotterObj.slider_widgets[1].GetRepresentation().SetValue(sliderThresholdValue)    # -> update upper threshold value
                self.currentSettings["upperThreshold"]=sliderThresholdValue     # also update upper threshold value in current settings
        except:                                                                 # other slider does not exists yet
            pass                                                                # do nothing


    #######################################################
    # Internal callback method for upper threshold slider #
    #######################################################
    def _callbackUpperThreshold(self,sliderThresholdValue):
        """Internal callback method for the upper threshold widget"""
        self.currentSettings["upperThreshold"]=sliderThresholdValue             # update upper threshold value with current slider position

        # prevent mixed up threshold sliders (upper < lower)
        try:                                                                    # use try-catch environment to prevent error during plot initialization (other slider is not defined yet)
            if sliderThresholdValue < self.plotterObj.slider_widgets[0].GetRepresentation().GetValue(): # check if value of upper threshold is below current lower threshold value
                self.plotterObj.slider_widgets[0].GetRepresentation().SetValue(sliderThresholdValue)    # -> update lower threshold value
                self.currentSettings["lowerThreshold"]=sliderThresholdValue     # also update lower threshold value in current settings
        except:                                                                 # other slider does not exist yet
            pass                                                                # do nothing


    ##################################################
    # Internal callback method for extraction widget #
    ##################################################
    def _callbackExtraction(self,boundsObj):
        """Internal callback method for the extraction box widget"""
        # update bounds of extraction box with current box bounds
        self.extractionBoxBounds.shallow_copy(boundsObj)                        # copy information of passed bounds object fom widget
        self.currentSettings["boxBounds"]=boundsObj.bounds                      # update current settings


    ##############################################
    # Internal method to define key press events #
    ##############################################
    def _keyPressEvents(self):
        """Internal method setting up user-defined key-press events"""
        if self.plotterObj.iren.GetKeySym() == "h":                             # check if "h"-key was pressed
            self.showCommandLineHelp()                                          # -> show help

        elif self.plotterObj.iren.GetKeySym() == "m":                           # check if "m"-key was pressed
            self.toggleMenu()                                                   # -> toggle menu

        elif self.plotterObj.iren.GetKeySym() == "d":                           # check if "d"-key was pressed
            self.restoreDefaultSettings()                                       # restore defaults

        elif self.plotterObj.iren.GetKeySym() == "space":                       # check if "space"-key was pressed
            self.filterPipeline()                                               # update filter pipeline with confirmed settings

        elif self.plotterObj.iren.GetKeySym() == "x":                           # check if "x"-key was pressed
            self.plotterObj.view_yz()                                           # set view to y-z-plane

        elif self.plotterObj.iren.GetKeySym() == "y":                           # check if "y"-key was pressed
            self.plotterObj.view_zx()                                           # set view to z-x-plane

        elif self.plotterObj.iren.GetKeySym() == "z":                           # check if "z"-key was pressed
            self.plotterObj.view_xy()                                           # set view to x-y-plane
