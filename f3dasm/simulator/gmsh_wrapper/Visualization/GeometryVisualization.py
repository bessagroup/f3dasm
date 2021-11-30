################################################################################
#              CLASS FOR GEOMETRY VISUALIZATION USING PYTHONOCC                #
################################################################################
# This file provides the class definition for the geometry visualization of the
# model. It is based on the pythonocc library and provides simple methods for
# the visualization.

############################
# Load required libraries: #
############################
# Standard Python libraries
import numpy as np                                                              # numpy for fast array computations
import tempfile as tf                                                           # tempfile for temporary files and folders
import logging                                                                  # logging for ẃarning messages
logger=logging.getLogger(__name__)                                              # -> set logger

# additional program libraries
try:                                                                            # try import of pythonocc library
    from OCC.Display.SimpleGui import init_display as initRenderer              # -> load rendering window initialization
    from OCC.Extend.DataExchange import read_step_file as stepReader            # step file reader
    from OCC.Core.TopoDS import TopoDS_Shape as topoShape                       # -> load creation of OCC shapes
    from OCC.Extend.TopologyUtils import TopologyExplorer as topoExplorer       # -> load topology exploring (needed to cycle through the objects of the compund)
    from OCC.Core.Quantity import Quantity_Color as color                       # -> load color methods
    from OCC.Core.Quantity import Quantity_TOC_RGB as typeRGB                   # -> load RGB color type
    from OCC.Core.Aspect import Aspect_TOTP_LEFT_LOWER                          # -> load property to place axes widget into the lower left corner
    from OCC.Core.V3d import V3d_ZBUFFER                                        # -> load property to place axes widget into the lower left corner
    PYTHONOCC_AVAILABLE=True                                                    # -> set availability flag to True
except ImportError:                                                             # handle unavailable OCC module
    PYTHONOCC_AVAILABLE=False                                                   # -> set availability flag to False
    logger.warning("The geometry visualization relies on the pythonocc package. To visualize the model geometry, install pythonocc.")


######################################
# Define GeometryVisualization class #
######################################
class GeometryVisualization():
    """Class for Gmsh model geometry visualization

    This class visualizes the geometry of a given GmshModel using the PYTHONOCC
    library. It provides a simple method to visualize the model geometry by
    storing the geometric model information into a temporary ".brep"-file which
    is read by PYTHONOCC.

    Attributes:
    -----------
    model: GmshModel object instance
        model instance for which the geometry has to be displayed

    renderer: renderer object instance
        instance of a rendering class

    groupColors: list of PYTHONOCC Quantity_Colors
        colors used for the visualization of the individual physical model groups

    lineColor: PYTHONOCC Quantity_Color
        color used for the cisualization of lines (especially in wireframe mode)
    """
    #########################
    # Initialization method #
    #########################
    def __init__(self,model=None):
        """Initilaization method for the GeometryVisualization class

        This method initializes a new instance of the GeometryVisualization
        class and visualizes the geometry of the passed GmshModel instance.
        Within the method, basic visualization settings as well as additional
        key press events for the rendering window are defined, before the
        visualization method is called.

        Parameters:
        model: GmshModel object instance
            model for which the geometry has to be visualized
        """
        # plausibility checks for input arguments
        if model is None:
            raise TypeError("No model instance to visualize passed. For the visualization of the model geometry, that model has to be known. Check your input data.")
        self.model=model

        # set renderer and place axes widget in the lower left corner
        self.renderer, self.startRenderer = initRenderer(display_triedron=False,background_gradient_color1=[82, 87, 110],background_gradient_color2=[82, 87, 110])[0:2]
        self.renderer.View.TriedronDisplay(Aspect_TOTP_LEFT_LOWER, color(1,1,1,typeRGB), 0.1, V3d_ZBUFFER)

        # set window title
        self.renderer.get_parent().parent().setWindowTitle("GmshModel Geometry Visualization")

        # set additional key events
        backendOpts=self.renderer.get_parent()                                  # get options of the rendering backend
        backendOpts._key_map.update({ord("X"): self.setViewX})                  # x-key pressed -> set view to y-z plane
        backendOpts._key_map.update({ord("Y"): self.setViewY})                  # y-key pressed -> set view to x-z plane
        backendOpts._key_map.update({ord("Z"): self.setViewZ})                  # z-key pressed -> set view to x-y plane
        backendOpts._key_map.update({ord("R"): self.resetView})                 # r-key pressed -> set reset view
        backendOpts._key_map.update({ord("Q"): self.close})                     # q-key pressed -> close window

        # set colors for the individual model groups
        self.groupColors=[]
        shapeColors=[color(0,0,0.7,typeRGB),                                    # (dark) blue
                     color(0.7,0,0,typeRGB),                                    # (dark) red
                     color(0,0.7,0,typeRGB),                                    # (dark) green
                     color(0.7,0,1,typeRGB),                                    # purple
                     color(1,0.57,0,typeRGB),                                   # orange
                     color(0,0.7,0.7,typeRGB),                                  # cyan
                     color(0.7,0.7,0.7,typeRGB),                                # (dark) grey
        ]
        self.lineColor=color(1,1,1,typeRGB)                                     # white
        for iEnt in model.gmshAPI.getEntities(model.dimension):                 # loop over all gmsh entities of the highest model dimension
            entGrp=model.gmshAPI.getPhysicalGroupsForEntity(*iEnt)              # get the physical group of the entity
            self.groupColors.append(shapeColors[entGrp[0]-1])                   # assign color assuming that order of shapes in brep file is the same as in Gmsh model

        # visualize the model geometry
        self.visualizeGeometry()



################################################################################
#                 MAIN METHOD FOR GEOMETRY VISUALIZATION                       #
################################################################################

    ####################################
    # Method to visualize the geometry #
    ####################################
    def visualizeGeometry(self):
        """Main method for the geometry visualization

        Within this method, the geometry of the GmshModel instance is visualized.
        A temporary geometry file is created and opened using the PYTHONOCC
        module.
        """
        with tf.TemporaryDirectory() as tmpDir:                                 # create temporary directory
            tmpFile=tmpDir+"/"+self.model.modelName+".step"                     # create name of file in temporary directory
            self.model.saveGeometry(tmpFile)                                    # write geometry to file in temporary directory

            # load geometry file with pythonocc
            compoundData=stepReader(tmpFile)                                    # read file and write compound object to compoundData

            # show all compound objects of the highest model dimension
            compoundTopology=topoExplorer(compoundData)                         # get shapes of the compound
            cntObj=0                                                            # initialize object counter for correct color coding of the objects
            for shape in self._getRelevantShape(compoundTopology):              # loop over all shapes
                obj=self.renderer.DisplayShape(shape,color=self.groupColors[cntObj],transparency = 0.05)[0] # -> visualize current shape
                self._setRenderingObjectProperties(obj)                         # -> set shape properties
                cntObj+=1                                                       # -> increase counter
            self.renderer.FitAll()                                              # fit view to rendering window
            self.startRenderer()                                                # activate/start rendering window



################################################################################
#               ADDITIONAL METHODS FOR SETTINGS AND KEY EVENTS                 #
################################################################################

    ###########################################
    # Method to set the view to the y-z-plane #
    ###########################################
    def setViewX(self):
        """Method to set the view to a plane with normal in x-direction"""
        self.renderer.View_Right()                                              # use pythonocc method to set correct view
        self.renderer.FitAll()                                                  # fit view in render window
        self.renderer.ZoomFactor(0.8)                                           # set zoom factor to add some space around the view


    ###########################################
    # Method to set the view to the x-z-plane #
    ###########################################
    def setViewY(self):
        """Method to set the view to a plane with normal in y-direction"""
        self.renderer.View_Rear()                                               # use pythonocc method to set correct view
        self.renderer.FitAll()                                                  # fit view in render window
        self.renderer.ZoomFactor(0.8)                                           # set zoom factor to add some space around the view


    ###########################################
    # Method to set the view to the x-y-plane #
    ###########################################
    def setViewZ(self):
        """Method to set the view to a plane with normal in z-direction"""
        self.renderer.View_Top()                                                # use pythonocc method to set correct view
        self.renderer.FitAll()                                                  # fit view in render window
        self.renderer.ZoomFactor(0.8)                                           # set zoom factor to add some space around the view


    ############################
    # Method to reset the view #
    ############################
    def resetView(self):
        """Method to reset the view"""
        self.renderer.View.Reset()                                              # use pythonocc method to reset view
        self.renderer.FitAll()                                                  # fit view in render window


    ########################################
    # Method to close the rendering window #
    ########################################
    def close(self):
        """Method to close the current rendering window"""
        self.renderer.get_parent().parent().close()                             # close parent rendering window



################################################################################
#           ADDITIONAL PRIVATE/HIDDEN METHODS FOR INTERNAL USE ONLY            #
################################################################################

    ############################################
    # Method to get the correct topology shape #
    ############################################
    def _getRelevantShape(self,topology):
        """Internal method to return the relevant shape method depending on the highest
        model dimension.

        Parameters:
        -----------
        topology: PYTHONOCC TopologyExplorer object instance
            topology for which the relevant shape has to be returned
        """
        if self.model.dimension==3:                                             # model dimension is 3
            return topology.solids()                                            # -> search for solids
        elif self.model.dimension==2:                                           # model dimension is 2
            return topology.shells()                                            # -> search for shells


    ###################################################
    # Method to configure rendering object properties #
    ###################################################
    def _setRenderingObjectProperties(self,obj):
        """Internal Method to set the rendering properties of a given object

        Parameters:
        -----------
        obj: PYTHONOCC AIS_SHAPE object instance
            rendering object to set the properties for
        """
        obj.GetContext().SetAutomaticHilight(False)                             # disable highlighting since this is only a viewer
        obj.SetWidth(3)                                                         # adjust line width
        attr=obj.Attributes()                                                   # get object attributes
        attr.SeenLineAspect().SetColor(self.lineColor)                          # set color of various types of lines to specified lineColor
        attr.HiddenLineAspect().SetColor(self.lineColor)
        attr.WireAspect().SetColor(self.lineColor)
        attr.LineAspect().SetColor(self.lineColor)
        attr.FaceBoundaryAspect().SetColor(self.lineColor)
        attr.FreeBoundaryAspect().SetColor(self.lineColor)
        attr.UnFreeBoundaryAspect().SetColor(self.lineColor)
