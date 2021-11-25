from gmshModel.Model import RandomInclusionRVE
from .domain import *
from .util import *
import numpy as np
import os 


class GENERATE_RVE():

    """

        RVE GENERATOR wrapper from gmshModel repository

    """

    def __init__(self, dim=2, Lc=4, r=0.3, directory=None, name='gmshModelRVE-collect', ext='.msh'):
        
        """ Initialize """

        self.dim = dim                      # Dimension of RVE 2 or 3 
        self.Lc = Lc                        # Size of the RVE
        self.r = r                          # size of the inclusions

        self.write_ext = ext                # Extension for saving 
        self.read_ext = '.xdmf'             # Extension for reading 

        
        if directory is None:
            self.directory_base = name
        else:
            self.directory_base = directory

        self.directory_base += '/'+str(dim)+'D/L:'+str(Lc)+'/r:'+str(r)


    def __call__(self,Vf,tag=None):              

        """ Call for the volume fraction of desire """

        self.directory = self.directory_base + '/Vf:'+str(int(Vf*100))

        self.init_Vf = Vf 

        try:

            self.create(tag)

            
        except KeyboardInterrupt:
            print(tag," is not created")

        self.extract_info()
        
        self.domain = DOMAIN(self.directory+self.read_ext)


    def create(self,tag,filename= "rve"):

        """ Method: Creation of RVE """ 

        ################################
        # Number of inclusion with the given radius
        ################################
        if self.dim == 2:
            self.inc_type = "Circle"
            self.size = [self.Lc, self.Lc, 0]        
            self.vol = self.Lc**2
            self.no = self.vol * self.init_Vf / (np.pi * self.r**2)
        
        elif self.dim == 3:
            self.inc_type = "Sphere"
            self.size = [self.Lc, self.Lc, self.Lc]
            self.vol = self.Lc**3
            self.no = self.vol * self.init_Vf / (4./3.*np.pi * self.r**3)

        self.set = [[self.r, np.rint(self.no)]] 

        ################################
        # gmshModel create and save RVE
        ################################
        self.initParameters = {                                                                
            "inclusionSets": self.set,
            "inclusionType": self.inc_type,
            "size": self.size,
            "origin": [0, 0, 0],                                                        
            "periodicityFlags": [1, 1, 1],                                              
            "domainGroup": "domain",                                                    
            "inclusionGroup": "inclusions",                                             
            "gmshConfigChanges": {"General.Terminal": 0,                                
                                  "General.Verbosity":4,
                                  "General.AbortOnError": 2,
                                  "Mesh.CharacteristicLengthExtendFromBoundary": 0}}
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

        if self.dim == 2:
            self.final_Vf = self.placed * np.pi * self.r**2 / self.vol
        else:
            self.final_Vf = self.placed * 4./3. * np.pi * self.r**3 / self.vol


        ################################
        # Directory Handling for the Vf
        ################################


        if tag != None:
            self.directory += '/'+str(tag)

        self.directory += '/'+filename

        self.RVE.saveMesh(self.directory+self.write_ext)

        self.RVE.close()

    def extract_info(self):

        """ Method: Extract all the information needed from xdmf file """

        xdmf_extract(self.directory+self.write_ext)


