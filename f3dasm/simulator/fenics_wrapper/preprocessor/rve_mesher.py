import numpy as np
import os

from f3dasm.simulator.gmsh_wrapper.Model import RandomInclusionRVE
from f3dasm.simulator.fenics_wrapper.preprocessor.gmsh_to_fenics import xdmf_extract

class RVEMesher():

    """
        RVE mesh generator
        Use gmsh to mesh the domain and fenics to generate domain file
    """

    def __init__(self, Lc=4, shape='Circle', size=[0.3]):
        """ 
        Initializes a class to generate RVEs
        
        Args:
            Lc (float): caracteristc length
            shape (string): shape of the includions. One of Circle, Sphere, Rectangle or Cylinder
            size (list of floats): size of the inclusions. For Circle or Sphere: [radius]; for Rectangle; [width, length]; for Cylinder: [radius], height will be Lc.
            directory (string): path to the base directory to store the outputs of mesh. Default is current directory.
            name (string): name of the directory where outputs will the stored. A directory with this name will be created in the base directory.
        
        Returns:
            Meshes for the RVE
        """

        self.Lc = Lc                        # Size of the RVE
        self.shape = shape                  # shape of the inclusion
        
        # Size and shape of the inclusions
        if self.shape=="Circle":
            self.r = size[0]
            self.dim = 2                    # Mesh dimension  
        elif self.shape=="Rectangle":
            self.w = size[0]
            self.l = size[1]
            self.dim = 2
        elif self.shape=='Sphere': 
            self.r = size[0]
            self.dim = 3                    # Mesh dimension 
        elif self.shape=='Cylinder':
            self.r = size[0]
            self.dim = 3
        else:
            raise NotImplementedError

        self.write_ext = '.msh'                # Extension for saving 
        self.read_ext = '.xdmf'             # Extension for reading 


    def mesh(self,Vf, work_dir, tag=None, max_mesh_size=''):              

        """ Call for the volume fraction of desire 
        tag (string): used to group rve-meshes in a single directory, one directory per tag
        max_mesh_size (float): maximun size for the mesh in the gmsModel. Defaul is one fifth 
        of the radius/length of the size of microstructures with 2 decimals precision.
        """
        if not os.path.exists(work_dir):
            os.mkdir(work_dir)
        self.work_dir = work_dir

        self.init_Vf = Vf # TODO: read from indata -> self.in_data.DATA.DataFrame.iloc[i]['Vf']
 
        try:
            # create mesh with gmsh
            self.create(tag)

        except KeyboardInterrupt:
            print(tag," is not created")

        # transform gmsh model to fenics domain
        self.extract_info()
        
        if isinstance(max_mesh_size, str):
            if self.shape == 'Rectangle':
                self.max_mesh_size = round(self.l/5, 2)
            else:
                self.max_mesh_size = round(self.r/5, 2)
        else:
            self.max_mesh_size = max_mesh_size

    def create(self,tag, filename= "rve"):

        """ Method: Creation of RVE """ 

        ###################################################
        # Number of inclusion with the given size and shape
        ###################################################
        
        # 2D Geometries
        if self.shape == "Circle":
            self.size = [self.Lc, self.Lc, 0]  # Size of the mesh
            self.vol = self.Lc**2
            self.no = self.vol * self.init_Vf / (np.pi * self.r**2)        
        elif self.shape == "Rectangle":
            self.size = [self.Lc, self.Lc, 0]  # Size of the mesh
            self.vol = self.Lc**2
            self.no = self.vol * self.init_Vf / (self.w * self.l)  
        # 3D Geometries
        elif self.shape == "Sphere":
            self.size = [self.Lc, self.Lc, self.Lc]
            self.vol = self.Lc**3
            self.no = self.vol * self.init_Vf / (4./3.*np.pi * self.r**3)
        elif self.shape == "Cylinder":
            self.size = [self.Lc, self.Lc, self.Lc]
            self.vol = self.Lc**3
            self.no = self.vol * self.init_Vf / (np.pi * self.r**2 * self.Lc)

        self.set = [[self.r, np.rint(self.no)]] 

        ################################
        # gmshModel create and save RVE
        ################################
        self.initParameters = {                                                                
            "inclusionSets": self.set,
            "inclusionType": self.shape,
            "size": self.size,
            "origin": [0, 0, 0],                                                        
            "periodicityFlags": [1, 1, 1],                                              
            "domainGroup": "domain",                                                    
            "inclusionGroup": "inclusions",                                             
            "gmshConfigChanges": {"General.Terminal": 0,                                
                                  "General.Verbosity":4,
                                  "General.AbortOnError": 2,
                                  "Mesh.CharacteristicLengthExtendFromBoundary": 0}}
        
        if self.shape == "Cylinder":
            self.initParameters["inclusionAxis"]= [0, 0, self.Lc] # define inclusionAxis direction. 
            # For now this must be pareller to the Z-axis and the same size as the RVE in that direction.
            # To allow other direction, the Gmsh module needs to be modified.

        self.modelingParameters = {                                                            
            "placementOptions": {"maxAttempts": 100000,                                  
                                 "minRelDistBnd": 0.01,                                  
                                 "minRelDistInc": 0.01}}

        self.RVE = RandomInclusionRVE(**self.initParameters) 

        self.RVE.createGmshModel(**self.modelingParameters)

        meshingParameters={                                                             
            "threads": None,                                                            
            "refinementOptions": {"maxMeshSize": 0.1,                                  
                                  "inclusionRefinement": True,                          
                                  "interInclusionRefinement": False,                    
                                  "elementsPerCircumference": 10,                       
                                  "elementsBetweenInclusions": 10,                       
                                  "inclusionRefinementWidth": 5,                        
                                  "transitionElements": "auto",                         
                                  "aspectRatio": 1.5}}

        self.RVE.createMesh(**meshingParameters)

        self.placed = np.sum(self.RVE.placementInfo)

        # Vf should be recomputed based on geometry shape
        if self.shape == "Circle":
            self.final_Vf = self.placed * np.pi * self.r**2 / self.vol
        elif self.shape == "Rectangle":
            self.final_Vf = self.placed * self.w * self.l / self.vol
        elif self.shape == "Sphere":
            self.final_Vf = self.placed * 4./3. * np.pi * self.r**3 / self.vol
        elif self.shape == "Cylinder":
            # TODO: How should this be computed under the current meshing limitations? a cylinder cross the z axis completely. 
            self.final_Vf = self.placed * np.pi * self.r**2 * self.Lc / self.vol
        else:
            raise NotImplementedError

        ################################
        # Handling Directory for the Vf
        ################################

        if tag != None:
            self.work_dir += '/'+str(tag)

        self.work_dir += '/'+filename

        self.RVE.saveMesh(self.work_dir+self.write_ext)

        self.RVE.close()

    def extract_info(self):

        """ Method: Extract all the information needed from xdmf file """
        
        xdmf_extract(self.work_dir+self.write_ext)

